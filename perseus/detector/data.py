import os
from dataclasses import dataclass
from operator import itemgetter
from typing import Iterator, Optional

import h5py
import numpy as np
import tifffile
import torch
from PIL import Image

# from super_gradients.common.decorators.factory_decorator import resolve_param
# from super_gradients.common.factories.transforms_factory import TransformsFactory
# from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import (
#     AbstractPoseEstimationDataset,
# )
# from super_gradients.training.samples import PoseEstimationSample
# from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform
from torch.utils.data import Dataset, DistributedSampler, Sampler

from perseus import ROOT


@dataclass(frozen=True)
class KeypointDatasetConfig:
    """Configuration for the keypoint dataset."""

    # dataset_path: str = "data/merged_lazy/merged.hdf5"
    dataset_path: str = "data/pruned_dataset/pruned.hdf5"
    lazy: bool = True


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
            self.weights = dataset["weights"][()]

            # always load these quantities into memory
            self.pixel_coordinates = torch.from_numpy(dataset["pixel_coordinates"][()])
            self.asset_ids = dataset["asset_ids"][()]  # used for segmentation images

            if cfg.lazy:
                self.image_filenames = dataset["image_filenames"][()]
                self.depth_filenames = dataset["depth_filenames"][()]
                self.segmentation_filenames = dataset["segmentation_filenames"][()]
            else:
                # depending on the dataset, these fields might not exist in the hdf5 file!
                self.images = dataset["images"][()][..., :3]
                self.depth_images = dataset["depth_images"][()]
                self.segmentation_images = dataset["segmentation_images"][()]

            # unused info in the dataset
            # self.object_poses = pp.SE3(torch.from_numpy(dataset["object_poses"][()]))
            # self.object_scales = torch.from_numpy(dataset["object_scales"][()])
            # self.camera_poses = pp.SE3(dataset["camera_poses"][()])
            # self.camera_intrinsics = torch.from_numpy(dataset["camera_intrinsics"][()])

    @property
    def num_trajectories(self) -> int:
        """The number of trajectories in the dataset."""
        if self.cfg.lazy:
            return len(self.image_filenames)
        else:
            return len(self.images)

    @property
    def images_per_trajectory(self) -> int:
        """The number of images per trajectory."""
        if self.cfg.lazy:
            return len(self.image_filenames[0])
        else:
            return len(self.images[0])

    def __len__(self) -> int:
        """The number of images in the dataset."""
        return self.num_trajectories * self.images_per_trajectory

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset."""
        traj_idx = idx // self.images_per_trajectory
        image_idx = idx % self.images_per_trajectory

        # loading the images
        if self.cfg.lazy:
            image_filename_local = self.image_filenames[traj_idx][image_idx].decode("utf-8")
            depth_filename_local = self.depth_filenames[traj_idx][image_idx].decode("utf-8")
            segmentation_filename_local = self.segmentation_filenames[traj_idx][image_idx].decode("utf-8")

            image_filename = str(os.path.join(ROOT, "data", image_filename_local))
            depth_filename = str(os.path.join(ROOT, "data", depth_filename_local))
            segmentation_filename = str(os.path.join(ROOT, "data", segmentation_filename_local))

            # _image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            _image = np.asarray(Image.open(image_filename).convert("RGB"), dtype=np.float32)  # can use PIL or opencv
            with tifffile.TiffFile(depth_filename) as tiff_file:
                _depth_image = tiff_file.pages[0].asarray()
            original_seg_image = np.asarray(Image.open(segmentation_filename))  # only PIL loads the channels correctly
        else:
            _image = self.images[traj_idx][image_idx]
            _depth_image = self.depth_images[traj_idx][image_idx]
            original_seg_image = self.segmentation_images[traj_idx][image_idx]

        # convert to tensor
        image = torch.from_numpy(_image.transpose(2, 0, 1) / 255.0)  # (C, H, W)
        depth_image = torch.from_numpy(_depth_image)

        # the segmentation image is a binary mask of the cube
        asset_id = self.asset_ids[traj_idx][image_idx]
        segmentation_image = np.zeros_like(original_seg_image)
        segmentation_image[np.array(original_seg_image) == (asset_id + 1)] = 1
        segmentation_image = torch.from_numpy(segmentation_image)

        pixel_coordinates = self.pixel_coordinates[traj_idx][image_idx]

        return {
            "image": image,
            "depth_image": depth_image,
            "segmentation_image": segmentation_image,
            "pixel_coordinates": pixel_coordinates,
            # "object_pose": self.object_poses[traj_idx][image_idx],
            # "camera_pose": self.camera_poses[traj_idx][image_idx],
            # "object_scale": self.object_scales[traj_idx][image_idx],
            # "camera_intrinsics": self.camera_intrinsics[traj_idx][image_idx],
            # "image_filename": self.image_filenames[traj_idx][image_idx],
            # "depth_filename": self.depth_filenames[traj_idx][image_idx],
            # "segmentation_filename": self.segmentation_filenames[traj_idx][image_idx],
        }


class PrunedKeypointDataset(Dataset):
    """A pruned keypoint dataset.

    The main difference between this dataset and the other type is that the data are pruned by a segmentation ratio.
    Because of this, trajectories might have different lengths, so in the pruned dataset, the data paths are flattened
    instead of being organized by trajectory. This means the implementation of __getitem__ is also affected.
    """

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
            self.weights = dataset["weights"][()]

            # always load these quantities into memory
            self.pixel_coordinates = torch.from_numpy(dataset["pixel_coordinates"][()])
            self.asset_ids = dataset["asset_ids"][()]  # used for segmentation images

            # the pruned dataset is always lazy
            self.image_filenames = dataset["image_filenames"][()]
            self.depth_filenames = dataset["depth_filenames"][()]
            self.segmentation_filenames = dataset["segmentation_filenames"][()]

    def __len__(self) -> int:
        """The number of images in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset."""
        image_filename_local = self.image_filenames[idx].decode("utf-8")
        depth_filename_local = self.depth_filenames[idx].decode("utf-8")
        segmentation_filename_local = self.segmentation_filenames[idx].decode("utf-8")

        image_filename = str(os.path.join(ROOT, "data", image_filename_local))
        depth_filename = str(os.path.join(ROOT, "data", depth_filename_local))
        segmentation_filename = str(os.path.join(ROOT, "data", segmentation_filename_local))

        _image = np.asarray(Image.open(image_filename).convert("RGB"), dtype=np.float32)  # can use PIL or opencv
        with tifffile.TiffFile(depth_filename) as tiff_file:
            _depth_image = tiff_file.pages[0].asarray()
        original_seg_image = np.asarray(Image.open(segmentation_filename))  # only PIL loads the channels correctly

        # convert to tensor
        image = torch.from_numpy(_image.transpose(2, 0, 1) / 255.0)  # (C, H, W)
        depth_image = torch.from_numpy(_depth_image)

        # the segmentation image is a binary mask of the cube
        asset_id = self.asset_ids[idx]
        segmentation_image = np.zeros_like(original_seg_image)
        segmentation_image[np.array(original_seg_image) == (asset_id + 1)] = 1
        segmentation_image = torch.from_numpy(segmentation_image)

        pixel_coordinates = self.pixel_coordinates[idx]

        return {
            "image": image,
            "depth_image": depth_image,
            "segmentation_image": segmentation_image,
            "pixel_coordinates": pixel_coordinates,
        }


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Stolen from
    https://github.com/catalyst-team/catalyst/blob/e99f90655d0efcf22559a46e928f0f98c9807ebf/catalyst/data/dataset.py#L6

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler) -> None:
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int) -> int:
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over `Sampler` for distributed training.

    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    Copy-pasted from
    https://github.com/catalyst-team/catalyst/blob/e99f90655d0efcf22559a46e928f0f98c9807ebf/catalyst/data/sampler.py#L499
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        """Initialize.

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


# ######## #
# YOLO-NAS #
# ######## #


# class KeypointDatasetYoloNas(AbstractPoseEstimationDataset):
#     """Dataset for keypoint detection for fine-tuning a YOLO-NAS model.

#     The super-gradients format is fairly particular, and they provide a trainer for fine-tuning their models, so we
#     create a dataset using their conventions to adapt things here.

#     Follows this tutorial:
#     https://github.com/Deci-AI/super-gradients/blob/master/notebooks/YoloNAS_Pose_Fine_Tuning_Animals_Pose_Dataset.ipynb
#     """

#     @resolve_param("transforms", TransformsFactory())
#     def __init__(
#         self,
#         data_dir: str,
#         transforms: Optional[List[AbstractKeypointTransform]] = None,
#         train: bool = True,
#         size: Optional[int] = None,
#         lazy: bool = True,
#         dataset_root: str | None = None,
#     ) -> None:
#         """Initialize the dataset.

#         Everything is in unnormalized coordinates in both the hdf5 file we generate as well as the YOLO-NAS format.

#         Args:
#             data_dir: The directory containing the hdf5 dataset.
#             transforms: The transforms to apply to the dataset.
#             train: Whether to load the training or test set.
#             size: The number of samples.
#             lazy: Whether to load the dataset lazily.
#             dataset_root: The root directory of the dataset (only used for lazy loading).
#         """
#         self.edge_links = [
#             [0, 1],
#             [0, 2],
#             [0, 4],
#             [1, 3],
#             [1, 5],
#             [2, 3],
#             [2, 6],
#             [3, 7],
#             [4, 5],
#             [4, 6],
#             [5, 7],
#             [6, 7],
#         ]
#         self.edge_colors = [
#             [31, 119, 180],
#             [174, 199, 232],
#             [255, 127, 14],
#             [255, 187, 120],
#             [44, 160, 44],
#             [152, 223, 138],
#             [214, 39, 40],
#             [255, 152, 150],
#             [148, 103, 189],
#             [197, 176, 213],
#             [140, 86, 75],
#             [196, 156, 148],
#         ]  # hardcoded colors from the tab20 colormap
#         self.keypoint_colors = [
#             [227, 119, 194],
#             [247, 182, 210],
#             [127, 127, 127],
#             [199, 199, 199],
#             [188, 189, 34],
#             [219, 219, 141],
#             [23, 190, 207],
#             [158, 218, 229],
#         ]  # hardcoded colors from the tab20 colormap
#         super().__init__(
#             transforms=transforms if transforms is not None else [],
#             num_joints=8,  # hardcoded to 8 for the cube asset
#             edge_links=self.edge_links,
#             edge_colors=self.edge_colors,
#             keypoint_colors=self.keypoint_colors,
#         )
#         self.lazy = lazy
#         self.dataset_root = dataset_root

#         with h5py.File(data_dir, "r") as f:
#             if train:
#                 dataset = f["train"]
#             else:
#                 dataset = f["test"]

#             if self.lazy:
#                 self.image_filenames = dataset["image_filenames"][()]
#                 self.const_mask = np.ones(dataset["images"][0].shape[-3:-1])  # (H, W, 3)
#             else:
#                 _images = dataset["images"][()]  # (num_videos, num_frames_per_video, H, W, 3)
#                 self.images = _images.reshape(-1, *_images.shape[-3:])  # (num_images, H, W, 3)
#                 self.const_mask = np.ones(self.images.shape[-3:-1])  # (H, W, 3)
#             _joints = dataset["pixel_coordinates"][()]
#             joints = _joints.reshape(-1, *_joints.shape[-2:])  # (num_images, 8, 2)
#             self.joints = np.concatenate(
#                 [joints, np.ones(joints.shape[:-1] + (1,))], axis=-1
#             )  # (num_images, 8, 3), all joints are "visible"
#             bboxes_x_min = np.min(self.joints[:, :, 0], axis=1)
#             bboxes_y_min = np.min(self.joints[:, :, 1], axis=1)
#             bboxes_x_max = np.max(self.joints[:, :, 0], axis=1)
#             bboxes_y_max = np.max(self.joints[:, :, 1], axis=1)
#             bboxes_w = bboxes_x_max - bboxes_x_min
#             bboxes_h = bboxes_y_max - bboxes_y_min
#             self.bboxes_xywh = np.stack([bboxes_x_min, bboxes_y_min, bboxes_w, bboxes_h], axis=1)  # (num_images, 4)

#             if size is not None:
#                 self.images = self.images[:size]
#                 self.joints = self.joints[:size]
#                 self.bboxes_xywh = self.bboxes_xywh[:size]

#     def __len__(self) -> int:
#         """The number of images in the dataset."""
#         if self.lazy:
#             return len(self.image_filenames)
#         else:
#             return len(self.images)

#     def load_sample(self, index: int) -> PoseEstimationSample:
#         """Load a sample from the dataset."""
#         if self.lazy:
#             if self.dataset_root is not None:
#                 img_path = os.path.join(self.dataset_root, self.image_filenames[index].decode("utf-8"))
#             else:
#                 img_path = self.image_filenames[index].decode("utf-8")
#             image = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         else:
#             image = self.images[index]
#         return PoseEstimationSample(
#             image=image,
#             mask=self.const_mask,
#             joints=self.joints[index][None, ...],
#             areas=None,
#             bboxes_xywh=self.bboxes_xywh[index][None, ...],
#             is_crowd=None,
#             additional_samples=None,
#         )


# if __name__ == "__main__":
#     cfg = tyro.cli(KeypointDatasetDebugConfig)
#     dataset = KeypointDataset(cfg.dataset_config)
#     augment = KeypointAugmentation(cfg.augmentation_config, train=cfg.train)

#     example = dataset[cfg.debug_index]
#     image = example["image"]
#     raw_pixel_coordinates = example["pixel_coordinates"]
#     pixel_coordinates = raw_pixel_coordinates.clone()

#     print(pixel_coordinates.shape, image.shape)

#     image, pixel_coordinates = augment(image.unsqueeze(0), pixel_coordinates.unsqueeze(0))

#     print("image augmented")

#     import matplotlib

#     matplotlib.use("Agg")
#     from matplotlib import pyplot as plt

#     # Visualize first image and keypoints.

#     image = image.squeeze(0)
#     pixel_coordinates = pixel_coordinates.reshape(-1, 2)

#     pixel_coordinates = kornia.geometry.conversions.denormalize_pixel_coordinates(
#         pixel_coordinates, image.shape[-2], image.shape[-1]
#     )

#     plt.imshow(image.permute(1, 2, 0).numpy())
#     plt.scatter(
#         pixel_coordinates[:, 0].numpy(),
#         pixel_coordinates[:, 1].numpy(),
#     )

#     plt.savefig("outputs/figures/test.png")
