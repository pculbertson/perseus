from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

from perseus import ROOT
from perseus.detector.data import KeypointDataset, KeypointDatasetConfig


def main(hdf5_path: Union[str, Path], mode: str = "train", img_type: str = "image") -> None:
    """Main function."""
    assert img_type in ["image", "depth_image", "segmentation_image"], f"Invalid img_type: {img_type}"
    num_keypoints = 8
    keypoint_colormap = plt.cm.get_cmap("tab10", num_keypoints)
    cfg = KeypointDatasetConfig(dataset_path=str(hdf5_path))
    dataset = KeypointDataset(cfg, train=True if mode == "train" else False)

    for i, example in enumerate(dataset):
        if img_type == "image":
            image = example["image"].permute(1, 2, 0)
        elif img_type == "depth_image":
            image = example["depth_image"]
        elif img_type == "segmentation_image":
            image = example["segmentation_image"]
        keypoints = example["pixel_coordinates"]
        # print(f"Max pixel coordinates: {keypoints.max()}")
        # print(f"Min pixel coordinates: {keypoints.min()}")

        plt.imshow(image)
        for k in range(len(keypoints)):
            plt.scatter(keypoints[k, 0], keypoints[k, 1], color=keypoint_colormap(k), marker="o")
        plt.title(f"Image {i}/{len(dataset)}")
        breakpoint()
        plt.show()


if __name__ == "__main__":
    hdf5_path = Path(f"{ROOT}/data/merged_lazy/merged.hdf5")

    # uncomment the one you want to see
    main(hdf5_path, mode="test", img_type="image")
    # main(hdf5_path, img_type="depth_image")
    # main(hdf5_path, img_type="segmentation_image")
