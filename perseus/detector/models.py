from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torchvision import models
from ultralytics.nn.autobackend import AutoBackend

from perseus import ROOT


class KeypointCNN(nn.Module):
    """Default perseus keypoint CNN model trained by fine-tuning resnet."""

    def __init__(self, n_keypoints: int = 8, num_channels: int = 3, H: int = 256, W: int = 256) -> None:
        """Initialize the keypoint CNN model.

        Args:
            n_keypoints: The number of keypoints to predict.
            num_channels: The number of channels in the input images.
            H: The height of the input images.
            W: The width of the input images.
        """
        super(KeypointCNN, self).__init__()
        # Load a prebuilt ResNet (e.g., ResNet18) and modify it
        self.resnet = models.resnet18(weights="DEFAULT")
        self.n_keypoints = n_keypoints
        self.num_channels = num_channels
        self.H = H
        self.W = W

        # Adjust the first convolutional layer if the input has a different number of channels than 3
        if num_channels != 3:  # noqa: PLR2004
            self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the average pooling and the final fully connected layer
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2 * n_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x: tensor of input images, shape=(batch_size, num_channels, H, W).
        """
        return self.resnet(x)


class KeypointGaussian(torch.nn.Module):
    """Keypoint CNN that returns a multivariate Gaussian over the keypoint locations in pixel coordinates."""

    def __init__(
        self,
        n_keypoints: int = 8,
        in_channels: int = 3,
        H: int = 256,
        W: int = 256,
        cov_type: str = "diag",
    ) -> None:
        """Initialize the keypoint CNN model.

        Args:
            n_keypoints: The number of keypoints to predict.
            in_channels: The number of channels in the input images.
            H: The height of the input images.
            W: The width of the input images.
            cov_type: The type of covariance matrix to use. Must be one of ["diag", "chol"].
        """
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
            self.fc3 = torch.nn.Linear(256, 2 * n_keypoints + (2 * n_keypoints * (2 * n_keypoints + 1) // 2))
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
            L += np.power(MIN_DET, 1 / (2 * self.n_keypoints)) * torch.eye(2 * self.n_keypoints, device=x.device)
            return mu, L

        elif self.cov_type == "diag":
            mu = x[:, : 2 * self.n_keypoints]
            MIN_COV = 1e-3
            sigma = torch.diag_embed(MIN_COV + torch.exp(x[:, 2 * self.n_keypoints :]))
            return mu, sigma
        else:
            raise ValueError(f"Invalid cov_type: {self.cov_type}")


class YOLOModel(nn.Module):
    """YOLO-based backbone for keypoint detection on the cube."""

    def __init__(self, version: int = 10, size: str = "n", n_keypoints: int = 8) -> None:
        """Initialize the YOLO model.

        Args:
            version: The version of YOLO to use. Must be 9 or 10.
            size: The size of the model to use.
                If v9, must be one of ["t", "s", "m", "c", "e"].
                If v10, must be one of ["n", "s", "m", "l", "x"].
            n_keypoints: The number of keypoints to predict.
        """
        super().__init__()

        # [DEBUG] for now, only allow YOLOv10
        assert version == 10, "[DEBUG] only allow v10 for now!"  # noqa: PLR2004

        # checking types
        if not isinstance(version, int):
            raise ValueError(f"Invalid version: {version}. Must be an integer.")
        if not isinstance(size, str):
            raise ValueError(f"Invalid size: {size}. Must be a string.")

        # checking values
        if version not in [9, 10]:
            raise ValueError(f"Invalid version: {version}. Must be one of [9, 10].")
        if version == 9 and size not in ["t", "s", "m", "c", "e"]:  # noqa: PLR2004
            raise ValueError(f"Invalid size: {size}. Must be one of ['t', 's', 'm', 'c', 'e'].")
        if version == 10 and size not in ["n", "s", "m", "l", "x"]:  # noqa: PLR2004
            raise ValueError(f"Invalid size: {size}. Must be one of ['n', 's', 'm', 'l', 'x'].")

        self.version = version
        self.n_keypoints = n_keypoints

        # loading yolo model
        model_str = ROOT + f"/outputs/yolo/yolov{version}{size}.pt"
        yolo_model = AutoBackend(model_str, device=torch.device("cuda"))  # ultralytics AutoBackend object
        self.yolo_detector = yolo_model.model  # ultralytics Detector object

        # keypoint head
        # The yolo model outputs 3 feature maps of shape (B, 144, 32/16/8, 32/16/8) for each of the one2many and
        # one2one computation branches. Both branches are used during training, but only the one2one branch is used
        # during inference.
        self.conv32x32_o2m = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=2 * n_keypoints, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv16x16_o2m = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=2 * n_keypoints, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv8x8_o2m = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=2 * n_keypoints, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv32x32_o2o = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=2 * n_keypoints, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv16x16_o2o = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=2 * n_keypoints, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv8x8_o2o = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=2 * n_keypoints, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.layer_weights = nn.Parameter(torch.ones(3))
        self.linear = nn.Linear(32 * 32, 1)  # TODO(ahl): pretty aggressive reduction here, might need to be adjusted

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the network.

        Args:
            x: tensor of input images, shape=(batch_size, 3, H, W).

        Returns:
            y_o2m: tensor of predicted pixel coordinates for the one2many branch, shape=(batch_size, 2 * n_keypoints).
            y_o2o: tensor of predicted pixel coordinates for the one2one branch, shape=(batch_size, 2 * n_keypoints).
        """
        # yolo part
        _detections, fpn_dicts = self.yolo_detector(x)  # [NOTE] detections unused
        o2m_features = fpn_dicts["one2many"]  # 3-tuple of features of shape (B, 144, 32/16/8, 32/16/8)
        o2o_features = fpn_dicts["one2one"]  # 3-tuple of features of shape (B, 144, 32/16/8, 32/16/8)

        # keypoint part
        y_o2o_32x32 = self.conv32x32_o2o(o2o_features[0])  # (B, 2 * n_keypoints, 32, 32)
        y_o2o_16x16 = F.interpolate(self.conv16x16_o2o(o2o_features[1]), scale_factor=2)  # (B, 2 * n_keypoints, 32, 32)
        y_o2o_8x8 = F.interpolate(self.conv8x8_o2o(o2o_features[2]), scale_factor=4)  # (B, 2 * n_keypoints, 32, 32)
        y_o2o = (
            self.layer_weights[0] * y_o2o_32x32
            + self.layer_weights[1] * y_o2o_16x16
            + self.layer_weights[2] * y_o2o_8x8
        ) / self.layer_weights.sum()  # (B, 2 * n_keypoints, 32, 32), learned weighted average of the three scales
        y_o2o = self.linear(y_o2o.view(*y_o2o.shape[:-2], -1)).squeeze(-1)

        # only do one2many computation during training
        if self.training:
            y_o2m_32x32 = self.conv32x32_o2m(o2m_features[0])  # (B, 2 * n_keypoints, 32, 32)
            y_o2m_16x16 = F.interpolate(self.conv16x16_o2m(o2m_features[1]), scale_factor=2)
            y_o2m_8x8 = F.interpolate(self.conv8x8_o2m(o2m_features[2]), scale_factor=4)
            y_o2m = (
                self.layer_weights[0] * y_o2m_32x32
                + self.layer_weights[1] * y_o2m_16x16
                + self.layer_weights[2] * y_o2m_8x8
            ) / self.layer_weights.sum()
            y_o2m = self.linear(y_o2m.view(*y_o2m.shape[:-2], -1)).squeeze(-1)  # (B, 2 * n_keypoints)
            return y_o2o, y_o2m

        return y_o2o
