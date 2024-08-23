from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt

from perseus import ROOT


def main(hdf5_path: Union[str, Path], mode: str = "images") -> None:
    """Main function."""
    assert mode in ["images", "depth_images", "segmentation_images"], f"Invalid mode: {mode}"
    num_keypoints = 8
    keypoint_colormap = plt.cm.get_cmap("tab10", num_keypoints)

    with h5py.File(hdf5_path, "r") as f:
        # max/min pixel coordinates over whole dataset
        max_pixel_coordinates = f["train"]["pixel_coordinates"][:].max()
        min_pixel_coordinates = f["train"]["pixel_coordinates"][:].min()
        print(f"Max pixel coordinates: {max_pixel_coordinates}")
        print(f"Min pixel coordinates: {min_pixel_coordinates}")

        # visualizing the training data
        for i in range(len(f["train"][f"{mode}"])):
            print(f"Image {i}")
            images = f["train"][f"{mode}"][i]
            keypoints_all = f["train"]["pixel_coordinates"][i]

            for j in range(len(images)):
                print(f"    Frame {j}")
                image = images[j]  # (256, 256, 3)
                keypoints = keypoints_all[j]  # (8, 2)

                # Plot the image and keypoints
                plt.imshow(image)
                for k in range(len(keypoints)):
                    plt.scatter(keypoints[k, 0], keypoints[k, 1], color=keypoint_colormap(k), marker="o")
                plt.title(f"Image {i}/{len(f['train'][f'{mode}'])} | Frame {j}/{len(images)}")
                plt.show()


if __name__ == "__main__":
    hdf5_path = Path(f"{ROOT}/data/merged/merged.hdf5")

    # uncomment the one you want to see
    main(hdf5_path, mode="images")
    # main(hdf5_path, mode="depth_images")
    # main(hdf5_path, mode="segmentation_images")
