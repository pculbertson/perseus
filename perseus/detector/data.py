import torch
from torch.utils.data import Dataset
import h5py
import pypose as pp
import numpy as np


class KeypointDataset(Dataset):
    def __init__(self, dataset_path, train=True, normalize=True):
        self.dataset_path = dataset_path
        self.train = train
        self.normalize = normalize

        # Load dataset.
        with h5py.File(dataset_path, "r") as f:
            if train:
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
            self.images = torch.from_numpy(self.dataset["images"][()])

            # Cast images to float.
            self.images = self.images.float()

            # Reorder images to (N, C, H, W).
            self.images = self.images.permute(0, 3, 1, 2)

            print(f"Images shape: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Normalize pixel coordinates.
        if self.normalize:
            pixel_coordinates = self.pixel_coordinates[idx].clone()
            pixel_coordinates[:, 0] = pixel_coordinates[:, 0] / (self.W / 2) - 1
            pixel_coordinates[:, 1] = pixel_coordinates[:, 1] / (self.H / 2) - 1
            pixel_coordinates = pixel_coordinates.reshape(-1)

        else:
            pixel_coordinates = self.pixel_coordinates[idx].reshape(-1)

        return (
            pixel_coordinates,
            self.object_poses[idx],
            self.images[idx],
        )
