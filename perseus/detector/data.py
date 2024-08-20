from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import kornia
import numpy as np
import pypose as pp
import torch
import tyro
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import (
    AbstractPoseEstimationDataset,
)
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform
from torch.utils.data import Dataset

from perseus import ROOT


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
    """Configuration for the keypoint dataset."""

    dataset_path: str = "data/qwerty_aggregated/mjc_data.hdf5"


@dataclass(frozen=True)
class KeypointDatasetDebugConfig:
    """Configuration for debugging the keypoint dataset."""

    dataset_config = KeypointDatasetConfig()
    augmentation_config = AugmentationConfig()
    debug_index: int = 0
    train: bool = True


class KeypointDataset(Dataset):
    """Dataset for keypoint detection."""

    def __init__(self, cfg: KeypointDatasetConfig, train: bool = True) -> None:
        """Initialize the dataset.

        Args:
            cfg: The dataset configuration.
            train: Whether to load the training or test set.
        """
        self.cfg = cfg
        self.train = train

        # Load dataset.
        if not cfg.dataset_path.startswith("/"):
            dataset_path = ROOT + f"/{cfg.dataset_path}"
        else:
            dataset_path = cfg.dataset_path
        with h5py.File(dataset_path, "r") as f:
            if self.train:
                dataset = f["train"]
            else:
                dataset = f["test"]

            self.W = f.attrs["W"]
            self.H = f.attrs["H"]

            self.pixel_coordinates = torch.from_numpy(dataset["pixel_coordinates"][()])
            self.object_poses = pp.SE3(torch.from_numpy(dataset["object_poses"][()]))
            self.images = dataset["images"][()][..., :3]
            self.object_scales = torch.from_numpy(dataset["object_scales"][()])
            self.camera_poses = pp.SE3(dataset["camera_poses"][()])
            self.camera_intrinsics = torch.from_numpy(dataset["camera_intrinsics"][()])
            self.image_filenames = dataset["image_filenames"][()]

            # print(f"Images shape: {self.images.shape}")

    @property
    def num_trajectories(self) -> int:
        """The number of trajectories in the dataset."""
        return len(self.images)

    @property
    def images_per_trajectory(self) -> int:
        """The number of images per trajectory."""
        return len(self.images[0])

    def __len__(self) -> int:
        """The number of images in the dataset."""
        return self.num_trajectories * self.images_per_trajectory

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset."""
        traj_idx = idx // self.images_per_trajectory
        image_idx = idx % self.images_per_trajectory

        image = kornia.utils.image_to_tensor(self.images[traj_idx][image_idx]) / 255.0
        pixel_coordinates = self.pixel_coordinates[traj_idx][image_idx]

        return {
            "image": image,
            "pixel_coordinates": pixel_coordinates,
            "object_pose": self.object_poses[traj_idx][image_idx],
            "camera_pose": self.camera_poses[traj_idx][image_idx],
            "object_scale": self.object_scales[traj_idx][image_idx],
            "camera_intrinsics": self.camera_intrinsics[traj_idx][image_idx],
            "image_filename": self.image_filenames[traj_idx][image_idx],
        }

    def get_trajectory(self, idx: int) -> dict:
        """Get a full trajectory from the dataset."""
        images = kornia.utils.image_to_tensor(self.images[idx]) / 255.0
        pixel_coordinates = self.pixel_coordinates[idx]

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
    """Data augmentation for keypoint detection."""

    def __init__(self, cfg: AugmentationConfig, train: bool = True) -> None:
        """Initialize the augmentation.

        Args:
            cfg: The augmentation configuration.
            train: Whether to apply training or test-time augmentations.
        """
        super().__init__()
        self.cfg = cfg
        self.train = train

        self.transforms = []

        if cfg.random_erasing:
            self.transforms.append(
                kornia.augmentation.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(2.0, 3.0), same_on_batch=False)
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
                kornia.augmentation.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(2.0, 3.0), same_on_batch=False)
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
            self.transforms.append(kornia.augmentation.RandomPlanckianJitter(mode="blackbody"))

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
            self.transforms.append(kornia.augmentation.RandomGaussianBlur((5, 5), (3.0, 8.0), p=0.5))

        self.transform_op = kornia.augmentation.AugmentationSequential(
            *self.transforms, data_keys=["image", "keypoints"]
        )

    def forward(self, images: torch.Tensor, pixel_coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to the images and pixel coordinates.

        Args:
            images: The images to augment.
            pixel_coordinates: The pixel coordinates to augment.

        Returns:
            images: The augmented images.
            pixel_coordinates: The augmented pixel coordinates.
        """
        B = images.shape[0]

        coords = pixel_coordinates.reshape(B, -1, 2)

        if len(self.transforms) > 0 and self.train:
            images, coords = self.transform_op(images, coords)

        coords = kornia.geometry.conversions.normalize_pixel_coordinates(coords, images.shape[-2], images.shape[-1])

        return images, coords.reshape(B, -1)


# ######## #
# YOLO-NAS #
# ######## #


class KeypointDatasetYoloNas(AbstractPoseEstimationDataset):
    """Dataset for keypoint detection for fine-tuning a YOLO-NAS model.

    The super-gradients format is fairly particular, and they provide a trainer for fine-tuning their models, so we
    create a dataset using their conventions to adapt things here.

    Follows this tutorial:
    https://github.com/Deci-AI/super-gradients/blob/master/notebooks/YoloNAS_Pose_Fine_Tuning_Animals_Pose_Dataset.ipynb
    """

    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        transforms: Optional[List[AbstractKeypointTransform]] = None,
        train: bool = True,
        size: Optional[int] = None,
    ) -> None:
        """Initialize the dataset.

        Everything is in unnormalized coordinates in both the hdf5 file we generate as well as the YOLO-NAS format.

        Args:
            data_dir: The directory containing the hdf5 dataset.
            transforms: The transforms to apply to the dataset.
            train: Whether to load the training or test set.
            size: The number of samples
        """
        self.edge_links = [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]
        self.edge_colors = [
            [31, 119, 180],
            [174, 199, 232],
            [255, 127, 14],
            [255, 187, 120],
            [44, 160, 44],
            [152, 223, 138],
            [214, 39, 40],
            [255, 152, 150],
            [148, 103, 189],
            [197, 176, 213],
            [140, 86, 75],
            [196, 156, 148],
        ]  # hardcoded colors from the tab20 colormap
        self.keypoint_colors = [
            [227, 119, 194],
            [247, 182, 210],
            [127, 127, 127],
            [199, 199, 199],
            [188, 189, 34],
            [219, 219, 141],
            [23, 190, 207],
            [158, 218, 229],
        ]  # hardcoded colors from the tab20 colormap
        super().__init__(
            transforms=transforms if transforms is not None else [],
            num_joints=8,  # hardcoded to 8 for the cube asset
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
        )

        with h5py.File(data_dir, "r") as f:
            if train:
                dataset = f["train"]
            else:
                dataset = f["test"]

            _images = dataset["images"][()]  # (num_videos, num_frames_per_video, H, W, 3)
            self.images = _images.reshape(-1, *_images.shape[-3:])  # (num_images, H, W, 3)
            self.const_mask = np.ones(self.images.shape[-3:-1])  # (H, W, 3)
            _joints = dataset["pixel_coordinates"][()]
            joints = _joints.reshape(-1, *_joints.shape[-2:])  # (num_images, 8, 2)
            self.joints = np.concatenate(
                [joints, np.ones(joints.shape[:-1] + (1,))], axis=-1
            )  # (num_images, 8, 3), all joints are "visible"
            bboxes_x_min = np.min(self.joints[:, :, 0], axis=1)
            bboxes_y_min = np.min(self.joints[:, :, 1], axis=1)
            bboxes_x_max = np.max(self.joints[:, :, 0], axis=1)
            bboxes_y_max = np.max(self.joints[:, :, 1], axis=1)
            bboxes_w = bboxes_x_max - bboxes_x_min
            bboxes_h = bboxes_y_max - bboxes_y_min
            self.bboxes_xywh = np.stack([bboxes_x_min, bboxes_y_min, bboxes_w, bboxes_h], axis=1)  # (num_images, 4)

            if size is not None:
                self.images = self.images[:size]
                self.joints = self.joints[:size]
                self.bboxes_xywh = self.bboxes_xywh[:size]

    def __len__(self) -> int:
        """The number of images in the dataset."""
        return len(self.images)

    def load_sample(self, index: int) -> PoseEstimationSample:
        """Load a sample from the dataset."""
        return PoseEstimationSample(
            image=self.images[index],
            mask=self.const_mask,
            joints=self.joints[index][None, ...],
            areas=None,
            bboxes_xywh=self.bboxes_xywh[index][None, ...],
            is_crowd=None,
            additional_samples=None,
        )


if __name__ == "__main__":
    cfg = tyro.cli(KeypointDatasetDebugConfig)
    dataset = KeypointDataset(cfg.dataset_config)
    augment = KeypointAugmentation(cfg.augmentation_config, train=cfg.train)

    example = dataset[cfg.debug_index]
    image = example["image"]
    raw_pixel_coordinates = example["pixel_coordinates"]
    pixel_coordinates = raw_pixel_coordinates.clone()

    print(pixel_coordinates.shape, image.shape)

    image, pixel_coordinates = augment(image.unsqueeze(0), pixel_coordinates.unsqueeze(0))

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
