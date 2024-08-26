import math
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from perseus import ROOT


def main(hdf5_path: str, lb: float = 0.02, ub: float = 0.5) -> None:
    """Visualize the segmentation ratios of the training set."""
    with h5py.File(hdf5_path, "r") as f:
        seg_ratios = f["train"]["segmentation_ratios"][()]

        # find the first images that belong in each of 100 bins for segmentation ratios
        representative_images = {}
        for i, _seg_ratio in enumerate(seg_ratios):
            for j, seg_ratio in enumerate(_seg_ratio):
                rounded_seg_ratio = math.floor(seg_ratio * 100) / 100  # round the seg_ratio into a bucket
                if rounded_seg_ratio not in representative_images and rounded_seg_ratio != 1.0:
                    segmentation_filename = f["train"]["segmentation_filenames"][i][j].decode("utf-8")
                    asset_id = f["train"]["asset_ids"][i][j]
                    segmentation_image = np.array(Image.open(os.path.join(ROOT, "data", segmentation_filename)))
                    representative_images[rounded_seg_ratio] = segmentation_image == asset_id + 1
                if len(representative_images) == 100:  # noqa: PLR2004
                    break

        seg_ratios = seg_ratios.flatten()  # for ease of plotting later, flatten

    # plot 1: histogram of segmentation ratios
    plt.hist(seg_ratios, bins=100, range=(0, 1))
    plt.title("Histogram of Segmentation Ratios")
    plt.xlabel("Segmentation Ratio")
    plt.ylabel("Frequency")
    plt.savefig("seg_ratios_hist.png")

    # plot 2: representative images by segmentation ratio
    plot_images = {}
    for ratio, image in representative_images.items():
        plot_images[ratio] = image

    fig, axs = plt.subplots(10, 10, figsize=(15, 15))
    axs = axs.flatten()

    sorted_buckets = sorted(plot_images.keys())
    for i, bucket in enumerate(sorted_buckets[:100]):
        image = 1 - plot_images[bucket]  # plot the negative for visibility with white background
        axs[i].imshow(image, cmap="gray")
        axs[i].set_title(f"Ratio: {bucket:.2f}", fontsize=8)
        axs[i].axis("off")

    plt.suptitle("Representative Images by Segmentation Ratio")
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig("seg_ratios_representative_images.png")

    # plot 3: cumulative distribution function of segmentation ratios
    plt.figure(figsize=(10, 6))
    sorted_ratios = np.sort(seg_ratios)
    cumulative_prob = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

    plt.plot(sorted_ratios, cumulative_prob, "b-")
    plt.title("Cumulative Distribution Function (CDF) of Segmentation Ratios")
    plt.xlabel("Segmentation Ratio")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("seg_ratios_cdf.png")

    # post-analysis text
    seg_ratios = np.array(seg_ratios)
    seg_ratios_filtered = seg_ratios[(seg_ratios >= lb) & (seg_ratios <= ub)]
    percentage_filtered = len(seg_ratios_filtered) / len(seg_ratios) * 100
    print(f"Percentage of data between {lb} and {ub} bins, inclusive: {percentage_filtered:.2f}%")

    plt.close()
    breakpoint()


if __name__ == "__main__":
    # NOTE: this only works on non-pruned datasets - it is meant to be a tool to inform how to prune
    hdf5_path = f"{ROOT}/data/merged_lazy/merged.hdf5"
    main(hdf5_path, lb=0.02, ub=0.9)
