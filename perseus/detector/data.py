import torch
from torch.utils.data import Dataset
import h5py
import kornia
import pypose as pp
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import tyro


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for data augmentation."""

    # Color jiggle parameters.
    brightness: float = 0.2
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.025

    # Random affine parameters.
    degrees: float = 90
    translate: Tuple[float, float] = (0.1, 0.1)
    scale: Tuple[float, float] = (0.9, 1.5)
    shear: float = 0.1

    # Flags for which augmentations to apply.
    color_jiggle: bool = True
    random_affine: bool = True
    planckian_jitter: bool = True
    random_erasing: bool = True
    blur: bool = True


@dataclass(frozen=True)
class KeypointDatasetConfig:
    dataset_path: str = "data/qwerty_aggregated/mjc_data.hdf5"


@dataclass(frozen=True)
class KeypointDatasetDebugConfig:
    dataset_config = KeypointDatasetConfig()
    augmentation_config = AugmentationConfig()
    debug_index: int = 0
    train: bool = True


class KeypointDataset(Dataset):
    def __init__(self, cfg: KeypointDatasetConfig, train=True):
        self.cfg = cfg
        self.train = train

        # Load dataset.
        with h5py.File(cfg.dataset_path, "r") as f:
            if self.train:
                self.dataset = f["train"]
            else:
                self.dataset = f["test"]

            self.W = f.attrs["W"]
            self.H = f.attrs["H"]

            self.pixel_coordinates = torch.from_numpy(
                self.dataset["pixel_coordinates"][()]
            )
            self.object_poses = pp.SE3(
                torch.from_numpy(self.dataset["object_poses"][()])
            )
            self.images = self.dataset["images"][()][..., :3]
            self.object_scales = torch.from_numpy(self.dataset["object_scales"][()])
            self.camera_poses = pp.SE3(self.dataset["camera_poses"][()])
            self.camera_intrinsics = torch.from_numpy(
                self.dataset["camera_intrinsics"][()]
            )
            self.image_filenames = self.dataset["image_filenames"][()]

            print(f"Images shape: {self.images.shape}")

    @property
    def num_trajectories(self):
        return len(self.images)

    @property
    def images_per_trajectory(self):
        return len(self.images[0])

    def __len__(self):
        return self.num_trajectories * self.images_per_trajectory

    def __getitem__(self, idx):
        traj_idx = idx // self.images_per_trajectory
        image_idx = idx % self.images_per_trajectory

        image = kornia.utils.image_to_tensor(self.images[traj_idx][image_idx]) / 255.0
        pixel_coordinates = self.pixel_coordinates[traj_idx][image_idx].clone()

        return {
            "image": image,
            "pixel_coordinates": pixel_coordinates,
            "object_pose": self.object_poses[traj_idx][image_idx],
            "camera_pose": self.camera_poses[traj_idx][image_idx],
            "object_scale": self.object_scales[traj_idx][image_idx],
            "camera_intrinsics": self.camera_intrinsics[traj_idx][image_idx],
            "image_filename": self.image_filenames[traj_idx][image_idx],
        }

    def get_trajectory(self, idx):
        images = kornia.utils.image_to_tensor(self.images[idx]) / 255.0
        pixel_coordinates = self.pixel_coordinates[idx].clone()

        return {
            "images": images,
            "pixel_coordinates": pixel_coordinates,
            "object_poses": self.object_poses[idx],
            "camera_poses": self.camera_poses[idx],
            "object_scales": self.object_scales[idx],
            "camera_intrinsics": self.camera_intrinsics[idx],
            "image_filenames": self.image_filenames[idx],
        }


class KeypointAugmentation(torch.nn.Module):
    def __init__(self, cfg: AugmentationConfig, train=True):
        super().__init__()
        self.cfg = cfg
        self.train = train

        self.transforms = []

        if cfg.random_erasing:
            self.transforms.append(
                kornia.augmentation.RandomErasing(
                    p=0.5, scale=(0.02, 0.1), ratio=(2.0, 3.0), same_on_batch=False
                )
            )
            self.transforms.append(
                kornia.augmentation.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.05),
                    ratio=(0.8, 1.2),
                    same_on_batch=False,
                    value=1,
                )
            )

        if cfg.random_affine:
            self.transforms.append(
                kornia.augmentation.RandomAffine(
                    degrees=cfg.degrees,
                    translate=cfg.translate,
                    scale=cfg.scale,
                    shear=cfg.shear,
                )
            )

        if cfg.random_erasing:
            self.transforms.append(
                kornia.augmentation.RandomErasing(
                    p=0.5, scale=(0.02, 0.1), ratio=(2.0, 3.0), same_on_batch=False
                )
            )
            self.transforms.append(
                kornia.augmentation.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.05),
                    ratio=(0.8, 1.2),
                    same_on_batch=False,
                    value=1,
                )
            )

        if cfg.planckian_jitter:
            self.transforms.append(
                kornia.augmentation.RandomPlanckianJitter(mode="blackbody")
            )

        if cfg.color_jiggle:
            self.transforms.append(
                kornia.augmentation.ColorJiggle(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                )
            )

        if cfg.blur:
            self.transforms.append(
                kornia.augmentation.RandomGaussianBlur((5, 5), (3.0, 8.0), p=0.5)
            )

        self.transform_op = kornia.augmentation.AugmentationSequential(
            *self.transforms, data_keys=["image", "keypoints"]
        )

    def forward(self, images, pixel_coordinates):
        B = images.shape[0]

        coords = pixel_coordinates.reshape(B, -1, 2)

        if len(self.transforms) > 0 and self.train:
            images, coords = self.transform_op(images, coords)

        coords = kornia.geometry.conversions.normalize_pixel_coordinates(
            coords, images.shape[-2], images.shape[-1]
        )

        return images, coords.reshape(B, -1)


if __name__ == "__main__":
    cfg = tyro.cli(KeypointDatasetDebugConfig)
    dataset = KeypointDataset(cfg.dataset_config)
    augment = KeypointAugmentation(cfg.augmentation_config, train=cfg.train)

    example = dataset[cfg.debug_index]
    image = example["image"]
    raw_pixel_coordinates = example["pixel_coordinates"]
    pixel_coordinates = raw_pixel_coordinates.clone()

    print(pixel_coordinates.shape, image.shape)

    image, pixel_coordinates = augment(
        image.unsqueeze(0), pixel_coordinates.unsqueeze(0)
    )

    print("image augmented")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Visualize first image and keypoints.

    image = image.squeeze(0)
    pixel_coordinates = pixel_coordinates.reshape(-1, 2)

    pixel_coordinates = kornia.geometry.conversions.denormalize_pixel_coordinates(
        pixel_coordinates, image.shape[-2], image.shape[-1]
    )

    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.scatter(
        pixel_coordinates[:, 0].numpy(),
        pixel_coordinates[:, 1].numpy(),
    )

    plt.savefig("outputs/figures/test.png")
