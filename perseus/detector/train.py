import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import wandb
from perseus.detector.data import AugmentationConfig, KeypointAugmentation, KeypointDataset, KeypointDatasetConfig
from perseus.detector.models import KeypointCNN, KeypointGaussian, YOLOModel
from wandb.util import generate_id

wandb.require("core")  # Use new wandb backend.


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training."""

    # The batch size.
    batch_size: int = 128

    # The (initial) learning rate set in the optimizer.
    learning_rate: float = 1e-3

    # The number of epochs to train for.
    n_epochs: int = 100

    # The device to train on.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # The number of workers to use for data loading.
    num_workers: int = -1

    # The output type of the model.
    output_type: str = "yolo"  # options: gaussian, regression, yolo

    # Training schedule.
    val_epochs: int = 1
    print_epochs: int = 1
    save_epochs: int = 5

    # Dataset parameters.
    dataset_config: KeypointDatasetConfig = KeypointDatasetConfig()

    # Data augmentation parameters.
    augmentation_config: AugmentationConfig = AugmentationConfig()

    # Model parameters.
    n_keypoints: int = 8
    in_channels: int = 3

    # Wandb config.
    wandb_project: str = "perseus-detector"


def train(cfg: TrainConfig) -> None:  # noqa: PLR0912, PLR0915
    """Train a keypoint detector model."""
    # Create dataloader.
    train_dataset = KeypointDataset(cfg.dataset_config, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    train_augment = KeypointAugmentation(cfg.augmentation_config, train=True)

    val_dataset = KeypointDataset(cfg.dataset_config, train=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_augment = KeypointAugmentation(
        cfg.augmentation_config, train=False
    )  # Still create this to do pixel coordinate conversion.

    # Initialize model and loss function
    if cfg.output_type == "gaussian":
        model = KeypointGaussian(cfg.n_keypoints, cfg.in_channels, train_dataset.H, train_dataset.W).to(cfg.device)

        if model.cov_type == "chol":

            def loss_fn(pred: torch.Tensor, pixel_coordinates: torch.Tensor) -> torch.Tensor:
                mu, L = pred

                return -torch.distributions.multivariate_normal.MultivariateNormal(mu, scale_tril=L).log_prob(
                    pixel_coordinates
                ).mean() - (cfg.n_keypoints / 2) * torch.log(torch.tensor(2 * np.pi))

        elif model.cov_type == "diag":

            def loss_fn(pred: torch.Tensor, pixel_coordinates: torch.Tensor) -> torch.Tensor:
                mu, sigma = pred

                return -torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma).log_prob(
                    pixel_coordinates
                ).mean() - (cfg.n_keypoints / 2) * torch.log(torch.tensor(2 * np.pi))

    elif cfg.output_type == "regression":
        model = KeypointCNN(cfg.n_keypoints, cfg.in_channels, train_dataset.H, train_dataset.W).to(cfg.device)
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    elif cfg.output_type == "yolo":
        model = YOLOModel(version=10, size="n", n_keypoints=8).to(cfg.device)
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        raise NotImplementedError(f"Output type {cfg.output_type} not implemented.")

    # Initialize optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5, verbose=True)

    # Initialize wandb.
    wandb_id = generate_id()
    wandb.init(
        project=cfg.wandb_project,
        config=cfg,
        id=wandb_id,
        resume="allow",
    )

    # Train model.
    for epoch in range(cfg.n_epochs):
        # Train model.
        model.train()
        for example in tqdm(train_dataloader, desc=f"Epoch {epoch}", total=len(train_dataloader)):
            images = example["image"]
            pixel_coordinates = example["pixel_coordinates"]

            # Move to device.
            pixel_coordinates = pixel_coordinates.to(cfg.device)
            images = images.to(cfg.device)

            # Augment data.
            images, pixel_coordinates = train_augment(images, pixel_coordinates)

            # Forward pass.
            if cfg.output_type == "yolo":
                # the yolo model uses a dual loss - we just add the one-to-many and one-to-one paths
                # [WARNING] this might make it hard to compare the train perf of the yolo model to the other models
                pred_o2o, pred_o2m = model(images)
                loss_o2o = loss_fn(pred_o2o, pixel_coordinates)
                loss_o2m = loss_fn(pred_o2m, pixel_coordinates)
                loss = loss_o2o + loss_o2m
            else:
                pred = model(images)

                # TODO(pculbert): add some validation / shape checking.

                # Compute loss.
                loss = loss_fn(pred, pixel_coordinates)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Log to wandb.
            wandb.log({"loss": loss.item()})

        if epoch % cfg.print_epochs == 0:
            print(f"Loss: {loss.item()}")

        if epoch % cfg.val_epochs == 0:
            # Validate model.
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for example in tqdm(val_dataloader, desc="Validation", total=len(val_dataloader)):
                    # Unpack example.
                    images = example["image"]
                    pixel_coordinates = example["pixel_coordinates"]

                    # Move to device.
                    pixel_coordinates = pixel_coordinates.to(cfg.device)
                    images = images.to(cfg.device)

                    # Augment data.
                    images, pixel_coordinates = val_augment(images, pixel_coordinates)

                    # Forward pass.
                    # [NOTE] when the model is in eval mode, the yolo model will only return the one-to-one path,
                    # so it becomes easier to compare the performance of the yolo model to the other models
                    pred = model(images)

                    # Compute loss.
                    loss = loss_fn(pred, pixel_coordinates)

                    val_loss += loss.item()

                val_loss /= len(val_dataloader)

                # Log to wandb.
                wandb.log({"val_loss": val_loss})
                print(f"Validation loss: {val_loss}")

                # Update learning rate.
                scheduler.step(val_loss)

        if epoch % cfg.save_epochs == 0:
            # Save model.
            # Create output directory if it doesn't exist.
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/{wandb_id}.pth")


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)
