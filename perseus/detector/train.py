import os
import re
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from perseus import ROOT
from perseus.detector.augmentations import AugmentationConfig, KeypointAugmentation
from perseus.detector.data import KeypointDatasetConfig, PrunedKeypointDataset
from perseus.detector.models import KeypointCNN
from wandb.util import generate_id

wandb.require("core")  # Use new wandb backend.


def rank_print(msg: str, rank: int = 0) -> None:
    """Prints only if rank is 0."""
    if rank == 0:
        print(msg)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training."""

    # The batch size.
    batch_size: int = 256

    # The (initial) learning rate set in the optimizer.
    learning_rate: float = 1e-3

    # The number of epochs to train for.
    n_epochs: int = 100

    # The device to train on.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # The number of workers to use for data loading.
    num_workers: int = -1

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
    in_channels: int = 4  # 3 for RGB, 4 for RGBD

    # Whether to use multi-gpu training
    multigpu: bool = True

    # If using multigpu, which gpu ids to use
    gpu_ids: str = re.sub(r"[\[\]\s]", "", str([_ for _ in range(torch.cuda.device_count())]))  # "0,1,2,..."

    # Whether to compile the model
    compile: bool = False

    # Whether to use automatic mixed precision
    amp: bool = True

    # Random seed
    random_seed: int = 42

    # Wandb config.
    wandb_project: str = "perseus-detector"


def initialize_training(  # noqa: PLR0912, PLR0915
    cfg: TrainConfig, rank: int = 0
) -> Tuple[
    torch.device,
    DataLoader,
    DataLoader,
    nn.Module,
    torch.optim.Optimizer,
    ReduceLROnPlateau,  # TODO(ahl): generalize the type
    Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    DistributedSampler,
    DistributedSampler,
    torch.cuda.amp.GradScaler,
    str,
]:
    """Initialize the training environment, especially (but not exclusively) for multi-gpu training.

    Args:
        cfg: The training configuration.
        rank: The rank of the process.
    """
    # set random seed
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # getting datasets
    train_dataset = PrunedKeypointDataset(cfg.dataset_config, train=True)
    val_dataset = PrunedKeypointDataset(cfg.dataset_config, train=False)

    # initializing the model + loss function
    model = KeypointCNN(cfg.n_keypoints, cfg.in_channels, train_dataset.H, train_dataset.W)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    # configuring setup based on whether we are using multi-gpu training or not
    if cfg.multigpu:
        print(f"Creating rank {rank} dataloaders...")

        num_gpus = cfg.gpu_ids.count(",") + 1  # gpu_ids is a comma-separated string of ids

        # initialize distributed training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=num_gpus)

        # each process gets its own device specified by the rank
        device = torch.device("cuda", rank)

        # distributed samplers associated with each process
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_gpus,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=num_gpus,
            rank=rank,
            shuffle=False,
        )
        train_shuffle = None
        val_shuffle = None

        # wrapping model in DDP + sending to device
        model = DDP(model.to(device), device_ids=[rank])

    else:
        print("Creating dataloaders...")

        # single gpu training only has one device
        device = torch.device(cfg.device)

        # no special samplers for single gpu training
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False

        # sending model to device
        model = model.to(device)

    # creating dataloaders
    num_workers = 4  # TODO(ahl): optimize this instead of using a magic number
    prefetch_factor = 2
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        sampler=train_sampler,
        multiprocessing_context="fork",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        sampler=val_sampler,
        multiprocessing_context="fork",
    )

    # checking for model compilation
    if cfg.compile:
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")  # compiled model

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.25, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # wandb - only log if rank is 0 (will be 0 by default for single-gpu training)
    wandb_id = generate_id()
    if rank == 0:
        wandb.init(project=cfg.wandb_project, config=cfg, id=wandb_id, resume="allow")

    return (
        device,
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_sampler,
        val_sampler,
        scaler,
        wandb_id,
    )


def train(cfg: TrainConfig, rank: int = 0) -> None:  # noqa: PLR0912, PLR0915
    """Train a keypoint detector model."""
    # Initialize training environment.
    (
        device,
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_sampler,
        _,  # val_sampler
        scaler,
        wandb_id,
    ) = initialize_training(cfg, rank)

    # Augmentation models.
    train_augment = KeypointAugmentation(cfg.augmentation_config, train=True)
    val_augment = KeypointAugmentation(cfg.augmentation_config, train=False)  # for pixel coordinate conversion

    # Main loop.
    for epoch in range(cfg.n_epochs):
        if cfg.multigpu:
            train_sampler.set_epoch(epoch)

        # Training loop.
        model.train()
        losses = []
        for example in tqdm(
            train_dataloader,
            desc=f"Iterations [Epoch {epoch}/{cfg.n_epochs}]",
            total=len(train_dataloader),
            leave=True,
            disable=(rank != 0),  # only rank 0 prints progress
        ):
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=cfg.amp
            ):
                images = example["image"].to(device)  # (B, 3, H, W)
                pixel_coordinates = example["pixel_coordinates"].to(device)

                # add a depth channel if needed
                if cfg.in_channels >= 4:  # noqa: PLR2004
                    depth_images = example["depth_image"].to(device)
                    images = torch.cat((images, depth_images[..., None, :, :]), dim=-3)  # (B, 4, H, W)

                # if doing the transplantation augmentation but not using the segmentation image in the model, we need
                # to add the segmentation image as an additional channel for just the augmentation and slice out the
                # rgbd channels afterwards
                if train_augment.cfg.random_transplantation_with_depth and cfg.in_channels < 5:  # noqa: PLR2004
                    seg_images = example["segmentation_image"].to(device)
                    if cfg.in_channels == 4:  # noqa: PLR2004
                        images_aug = torch.cat((images, seg_images[..., None, :, :]), dim=-3)
                    else:
                        images_aug = torch.cat(
                            (images, depth_images[..., None, :, :], seg_images[..., None, :, :]), dim=-3
                        )
                else:
                    images_aug = images
                _images, pixel_coordinates = train_augment(images_aug, pixel_coordinates)
                images = _images[..., : cfg.in_channels, :, :]  # (B, in_channels, H, W)

                # Forward pass.
                pred = model(images)
                if pred.shape[-1] != 2:  # noqa: PLR2004
                    pred = pred.reshape(*pred.shape[:-1], cfg.n_keypoints, 2)
                loss = loss_fn(pred, pixel_coordinates)

            # Log loss.
            losses.append(loss.item())
            if (not cfg.multigpu) or rank == 0:
                wandb.log({"loss": loss.item()})

            # Backward pass.
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip grads
            scaler.step(optimizer)
            scaler.update()

        if epoch % cfg.print_epochs == 0:
            rank_print(f"    Avg. Loss in Epoch: {np.mean(losses)}", rank=rank)

        # Validation loop.
        if epoch % cfg.val_epochs == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for example in tqdm(
                    val_dataloader,
                    desc="Validation",
                    total=len(val_dataloader),
                    leave=True,
                    disable=(rank != 0),  # only rank 0 prints progress
                ):
                    with torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=cfg.amp
                    ):
                        images = example["image"].to(device)
                        pixel_coordinates = example["pixel_coordinates"].to(device)
                        if cfg.in_channels == 4:  # noqa: PLR2004
                            depth_images = example["depth_image"].to(device)
                            images = torch.cat((images, depth_images[..., None, :, :]), dim=-3)
                        images, pixel_coordinates = val_augment(images, pixel_coordinates)

                        # Forward pass.
                        # [NOTE] when the model is in eval mode, the yolo model will only return the one-to-one path,
                        # so it becomes easier to compare the performance of the yolo model to the other models
                        pred = model(images)
                        if pred.shape[-1] != 2:  # noqa: PLR2004
                            pred = pred.reshape(*pred.shape[:-1], cfg.n_keypoints, 2)
                        loss = loss_fn(pred, pixel_coordinates)

                    val_loss += loss.item()

                val_loss /= len(val_dataloader)

                # Log to wandb.
                if (not cfg.multigpu) or rank == 0:
                    wandb.log({"val_loss": val_loss})
                rank_print(f"    Validation loss: {val_loss}", rank=rank)

                # Update learning rate based on val loss.
                scheduler.step(val_loss)

        # Model saving.
        if epoch % cfg.save_epochs == 0:
            os.makedirs("outputs/models", exist_ok=True)  # Create output directory if it doesn't exist.
            if rank == 0:  # models are synced, but this avoids saving the same model num_gpus times
                torch.save(model.state_dict(), f"{ROOT}/outputs/models/{wandb_id}.pth")

    # Cleanup.
    if cfg.multigpu:
        dist.destroy_process_group()


def _train_multigpu(rank: int, cfg: TrainConfig) -> None:
    """Trivial wrapper for train.

    Defined this way because rank must be the first argument in the function signature for mp.spawn.
    This function must also be defined at a module top level to allow pickling.
    """
    train(cfg, rank=rank)


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    if cfg.multigpu:
        mp.spawn(_train_multigpu, args=(cfg,), nprocs=cfg.gpu_ids.count(",") + 1, join=True)
    else:
        train(cfg, rank=0)
