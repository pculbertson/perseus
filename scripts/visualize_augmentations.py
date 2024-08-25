import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch

from perseus import ROOT
from perseus.detector.augmentations import AugmentationConfig, KeypointAugmentation
from perseus.detector.data import KeypointDataset, KeypointDatasetConfig, PrunedKeypointDataset


def visualize_dataset(dataset: KeypointDataset, augmentation: KeypointAugmentation, mode: str) -> None:
    """Visualize a dataset."""
    # Draw 16 random examples from the dataset
    inds = np.random.choice(len(dataset), 16)
    images_stacked = []
    pixel_coordinates = []
    for i in inds:
        example = dataset[i]
        image = example["image"]
        depth_image = example["depth_image"][None, ...]
        segmentation_image = example["segmentation_image"][None, ...]
        pixel_coords = example["pixel_coordinates"]

        image_stacked = np.concatenate([image, depth_image, segmentation_image], axis=-3)
        image_stacked, pixel_coords = augmentation(torch.tensor(image_stacked), pixel_coords)

        images_stacked.append(image_stacked.numpy()[0].transpose(1, 2, 0))
        pixel_coordinates.append(
            kornia.geometry.denormalize_pixel_coordinates(pixel_coords, dataset.H, dataset.W).cpu().numpy()
        )

    # Create a square plot, with each entry being a 1x3 subplot
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)  # Adjust size and DPI as needed
    rows, cols = 4, 4
    axes = fig.subplots(rows, cols)
    for idx, ax in enumerate(axes.flat):
        if idx < len(images_stacked):
            # Create a 1x3 subplot for each image, depth image, and segmentation image
            img_ax = ax.inset_axes([0, 0, 0.33, 1])
            depth_ax = ax.inset_axes([0.33, 0, 0.33, 1])
            seg_ax = ax.inset_axes([0.66, 0, 0.33, 1])

            img_ax.imshow(images_stacked[idx][..., :3])
            img_ax.axis("off")

            depth_ax.imshow(images_stacked[idx][..., 3], cmap="gray")
            depth_ax.axis("off")

            seg_ax.imshow(images_stacked[idx][..., 4], cmap="gray")
            seg_ax.axis("off")

            # Plot the keypoints on each image
            for k in range(len(pixel_coordinates[idx])):
                img_ax.scatter(pixel_coordinates[idx][k, 0], pixel_coordinates[idx][k, 1], color="red", marker="o")
                depth_ax.scatter(pixel_coordinates[idx][k, 0], pixel_coordinates[idx][k, 1], color="red", marker="o")
                seg_ax.scatter(pixel_coordinates[idx][k, 0], pixel_coordinates[idx][k, 1], color="red", marker="o")

            ax.axis("off")

    plt.suptitle(f"{mode} Data Augmentation Examples")
    plt.tight_layout()
    plt.savefig(f"{mode.lower()}_data_aug_examples.png", bbox_inches="tight", pad_inches=0)


def main(hdf5_path: str, pruned: bool = True) -> None:
    """Main function."""
    data_cfg = KeypointDatasetConfig(dataset_path=hdf5_path)

    # Load train and test datasets
    if pruned:
        train_dataset = PrunedKeypointDataset(data_cfg, train=True)
        test_dataset = PrunedKeypointDataset(data_cfg, train=False)
    else:
        train_dataset = KeypointDataset(data_cfg, train=True)
        test_dataset = KeypointDataset(data_cfg, train=False)

    aug_cfg = AugmentationConfig()
    train_augmentation = KeypointAugmentation(aug_cfg, train=True)
    test_augmentation = KeypointAugmentation(aug_cfg, train=False)

    # Visualize train and test datasets
    visualize_dataset(train_dataset, train_augmentation, mode="Train")
    visualize_dataset(test_dataset, test_augmentation, mode="Test")


if __name__ == "__main__":
    # hdf5_path = f"{ROOT}/data/merged_lazy/merged.hdf5"
    hdf5_path = f"{ROOT}/data/pruned_dataset/pruned.hdf5"
    main(hdf5_path, pruned=True)
