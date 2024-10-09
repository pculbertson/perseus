import os
from dataclasses import dataclass

import h5py
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

from perseus import ROOT


@dataclass(frozen=True)
class KeypointDatasetConfig:
    """Configuration for the keypoint dataset."""

    dataset_path: str = "data/pruned_dataset/pruned.hdf5"
    lazy: bool = True


class PrunedKeypointDataset(Dataset):
    """A pruned keypoint dataset.

    When the data are initially generated, they are organized by trajectory (i.e., the image sequence corresponding to a
    single rendered rollout). However, when the data are pruned by a segmentation ratio, the number of images per
    trajectory is no longer known beforehand, so the images must all be flattened.
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
