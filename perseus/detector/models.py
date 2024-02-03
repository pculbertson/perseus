import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class KeypointCNN(nn.Module):
    def __init__(self, n_keypoints=8, num_channels=3, H=256, W=256):
        super(KeypointCNN, self).__init__()
        # Load a prebuilt ResNet (e.g., ResNet18) and modify it
        self.resnet = models.resnet18(weights="DEFAULT")
        self.n_keypoints = n_keypoints
        self.num_channels = num_channels
        self.H = H
        self.W = W

        # Adjust the first convolutional layer if the input has a different number of channels than 3
        if num_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace the average pooling and the final fully connected layer
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2 * n_keypoints)

    def forward(self, x):
        return self.resnet(x)


class KeypointGaussian(torch.nn.Module):
    """Implementation of a keypoint CNN that returns a multivariate Gaussian over the keypoint locations in pixel coordinatess."""

    def __init__(
        self,
        n_keypoints: int = 8,
        in_channels: int = 3,
        H: int = 256,
        W: int = 256,
        cov_type: str = "diag",
    ):
        super().__init__()

        self.n_keypoints = n_keypoints
        self.in_channels = in_channels
        self.H = H
        self.W = W

        # (batch_size, 64, H/2, W/2)
        self.conv1 = torch.nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(128)

        self.fc1 = torch.nn.Linear(128 * (H // 32) * (W // 32), 256)
        self.fc2 = torch.nn.Linear(256, 256)

        if cov_type == "chol":
            self.fc3 = torch.nn.Linear(
                256, 2 * n_keypoints + (2 * n_keypoints * (2 * n_keypoints + 1) // 2)
            )
        elif cov_type == "diag":
            self.fc3 = torch.nn.Linear(256, 2 * 2 * n_keypoints)

        self.relu = torch.nn.ReLU()

        self.cov_type = cov_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x: (batch_size, in_channels, H, W) tensor of input images.

        Returns:
            (batch_size, 2 * n_keypoints) tensor of predicted pixel coordinates.
        """

        # (batch_size, 64, H/2, W/2)
        x = self.relu(self.bn1(self.conv1(x)))

        # (batch_size, 128, H/4, W/4)
        x = self.relu(self.bn2(self.conv2(x)))

        # (batch_size, 256, H/8, W/8)
        x = self.relu(self.bn3(self.conv3(x)))

        # (batch_size, 512, H/16, W/16)
        x = self.relu(self.bn4(self.conv4(x)))

        # (batch_size, 512, H/32, W/32)
        x = self.relu(self.bn5(self.conv5(x)))

        # (batch_size, 512 * H/32 * W/32)
        x = torch.flatten(x, start_dim=1)

        # (batch_size, 1024)
        x = self.relu(self.fc1(x))

        # (batch_size, 1024)
        x = self.relu(self.fc2(x))

        # (batch_size, 2 * n_keypoints)
        x = self.fc3(x)

        if self.cov_type == "chol":
            mu = x[:, : 2 * self.n_keypoints]
            L = torch.zeros(
                (x.shape[0], 2 * self.n_keypoints, 2 * self.n_keypoints),
                device=x.device,
            )

            # Compute the lower triangular matrix.
            rows, cols = torch.tril_indices(2 * self.n_keypoints, 2 * self.n_keypoints)
            L[:, rows, cols] = x[:, 2 * self.n_keypoints :]

            # Exponentiate the diagonal.
            L[
                :,
                torch.arange(2 * self.n_keypoints),
                torch.arange(2 * self.n_keypoints),
            ] = torch.exp(
                L[
                    :,
                    torch.arange(2 * self.n_keypoints),
                    torch.arange(2 * self.n_keypoints),
                ]
            )

            MIN_DET = 1e-8  # Minimum determinant of the covariance matrix.
            L += np.power(MIN_DET, 1 / (2 * self.n_keypoints)) * torch.eye(
                2 * self.n_keypoints, device=x.device
            )
            return mu, L

        elif self.cov_type == "diag":
            mu = x[:, : 2 * self.n_keypoints]
            MIN_COV = 1e-3
            sigma = torch.diag_embed(MIN_COV + torch.exp(x[:, 2 * self.n_keypoints :]))
            return mu, sigma
        else:
            raise ValueError(f"Invalid cov_type: {self.cov_type}")
