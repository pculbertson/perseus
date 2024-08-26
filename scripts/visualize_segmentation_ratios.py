import math
import os
from multiprocessing import Pool, cpu_count

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from perseus import ROOT


def process_image_set(args: tuple) -> tuple | None:
    """Process an image set and return the segmentation ratio."""
    sfp, asset_id = args
    try:
        segmentation_path = os.path.join(ROOT, "data", sfp)
        with Image.open(segmentation_path) as segmentation:
            segmentation_np = np.array(segmentation)

        asset_id_plus_one = asset_id + 1
        seg_mask = segmentation_np == asset_id_plus_one
        seg_ratio = np.mean(seg_mask)
        return seg_ratio, seg_mask
    except Exception as e:
        print(f"Error processing {sfp}, {asset_id}: {e}")
        return None


def aggregate_results(results: list, shape: tuple) -> tuple:
    """Aggregate the results of the image set processing."""
    seg_ratios = np.full(shape, np.nan)
    representative_images = {}

    index = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if results[index] is not None:
                seg_ratio, segmentation = results[index]
                seg_ratios[i, j] = seg_ratio
                if seg_ratio not in representative_images:
                    representative_images[seg_ratio] = segmentation
            index += 1

    return seg_ratios, representative_images


def main(hdf5_path: str, lb: float = 0.02, ub: float = 0.5) -> None:
    """Visualize the segmentation ratios of the training set."""
    with h5py.File(hdf5_path, "r") as f:
        train = f["train"]
        segmentation_filenames = train["segmentation_filenames"][()]
        asset_ids = train["asset_ids"][()]

        args_list = [
            (sfp.decode("utf-8") if isinstance(sfp, bytes) else sfp, asset_id)
            for _sfp, _asset_id in zip(segmentation_filenames, asset_ids, strict=False)
            for sfp, asset_id in zip(_sfp, _asset_id, strict=False)
        ]

        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_image_set, args_list, chunksize=10), total=len(args_list)))
            seg_ratios, representative_images = aggregate_results(results, segmentation_filenames.shape)

    seg_ratios = seg_ratios.flatten()

    # plot 1: histogram of segmentation ratios
    plt.hist(seg_ratios, bins=100, range=(0, 1))
    plt.title("Histogram of Segmentation Ratios")
    plt.xlabel("Segmentation Ratio")
    plt.ylabel("Frequency")
    plt.savefig("seg_ratios_hist.png")

    # plot 2: representative images by segmentation ratio
    plot_images = {}
    for ratio, image in representative_images.items():
        bucket = math.floor(ratio * 100) / 100
        if bucket not in plot_images:
            plot_images[bucket] = image
        if len(plot_images) == 101:  # noqa: PLR2004
            break

    fig, axs = plt.subplots(10, 10, figsize=(15, 15))
    axs = axs.flatten()

    sorted_buckets = sorted(plot_images.keys())
    for i, bucket in enumerate(sorted_buckets[:100]):
        image = 1 - plot_images[bucket]
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
