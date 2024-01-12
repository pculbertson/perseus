from dataclasses import dataclass
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import tyro
from wandb.util import generate_id
import wandb

from perseus.detector.models import KeypointCNN, KeypointGaussian
from perseus.detector.data import KeypointDataset


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training."""

    # Training parameters.
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 1

    output_type: str = "regression"

    val_epochs: int = 3
    print_epochs: int = 1
    save_epochs: int = 5

    # Dataset parameters.
    dataset_path: str = "data/2023-12-28_20-34-56/mjc_data.hdf5"

    # Model parameters.
    n_keypoints: int = 8
    in_channels: int = 3

    # Wandb config.
    wandb_project: str = "perseus-detector"


def train(cfg: TrainConfig):
    # Create dataloader.
    train_dataset = KeypointDataset(cfg.dataset_path, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    val_dataset = KeypointDataset(cfg.dataset_path, train=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # Initialize model.
    if cfg.output_type == "gaussian":
        model = KeypointGaussian(
            cfg.n_keypoints, cfg.in_channels, train_dataset.H, train_dataset.W
        ).to(cfg.device)
    elif cfg.output_type == "regression":
        model = KeypointCNN(
            cfg.n_keypoints, cfg.in_channels, train_dataset.H, train_dataset.W
        ).to(cfg.device)

    # Initialize optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Initialize loss function.
    if cfg.output_type == "gaussian":
        if model.cov_type == "chol":

            def loss_fn(pred, pixel_coordinates):
                mu, L = pred

                return -torch.distributions.multivariate_normal.MultivariateNormal(
                    mu, scale_tril=L
                ).log_prob(pixel_coordinates).mean() - (
                    cfg.n_keypoints / 2
                ) * torch.log(
                    torch.tensor(2 * np.pi)
                )

        elif model.cov_type == "diag":

            def loss_fn(pred, pixel_coordinates):
                mu, sigma = pred

                return -torch.distributions.multivariate_normal.MultivariateNormal(
                    mu, sigma
                ).log_prob(pixel_coordinates).mean() - (
                    cfg.n_keypoints / 2
                ) * torch.log(
                    torch.tensor(2 * np.pi)
                )

    elif cfg.output_type == "regression":
        loss_fn = nn.SmoothL1Loss(beta=0.25)

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
        for i, (pixel_coordinates, object_poses, images) in tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
        ):
            # Move to device.
            pixel_coordinates = pixel_coordinates.to(cfg.device)
            object_poses = object_poses.to(cfg.device)
            images = images.to(cfg.device)

            # Forward pass.
            pred = model(images)

            # TODO(pculbert): add some validation / shape checking.

            # Compute loss.
            loss = loss_fn(pred, pixel_coordinates)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

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
                for i, (pixel_coordinates, object_poses, images) in tqdm(
                    enumerate(val_dataloader),
                    desc=f"Validation",
                    total=len(val_dataloader),
                ):
                    # Move to device.
                    pixel_coordinates = pixel_coordinates.to(cfg.device)
                    object_poses = object_poses.to(cfg.device)
                    images = images.to(cfg.device)

                    # Forward pass.
                    pred = model(images)

                    # Compute loss.
                    loss = loss_fn(pred, pixel_coordinates)

                    val_loss += loss.item()

                val_loss /= len(val_dataloader)

                # Log to wandb.
                wandb.log({"val_loss": val_loss})
                print(f"Validation loss: {val_loss}")

        if epoch % cfg.save_epochs == 0:
            # Save model.
            # Create output directory if it doesn't exist.
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/{wandb_id}.pth")


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)
