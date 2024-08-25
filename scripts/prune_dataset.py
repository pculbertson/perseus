import multiprocessing as mp
import os
import shutil

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from perseus import ROOT


def calculate_segmentation_ratio(seg_filename: str, asset_id: int) -> float:
    """Calculate the segmentation ratio for a given segmentation file and asset ID."""
    seg_path = os.path.join(ROOT, "data", seg_filename)
    with Image.open(seg_path) as segmentation:
        segmentation_np = np.array(segmentation)

    seg_mask = segmentation_np == (asset_id + 1)
    seg_ratio = np.mean(seg_mask)
    return seg_ratio


def create_args_list(dataset_in: h5py.Group, split: str, lb: float, ub: float, output_data_dir: str) -> list:
    """Create a list of arguments for processing images."""
    total_images = sum(len(traj) for traj in dataset_in["image_filenames"])
    args_list = []
    current_index = 0

    with tqdm(total=total_images, desc=f"Creating args list for {split}") as pbar:
        for traj_idx in range(len(dataset_in["image_filenames"])):
            traj_length = len(dataset_in["image_filenames"][traj_idx])

            image_filenames = dataset_in["image_filenames"][traj_idx]
            depth_filenames = dataset_in["depth_filenames"][traj_idx]
            segmentation_filenames = dataset_in["segmentation_filenames"][traj_idx]
            pixel_coordinates = dataset_in["pixel_coordinates"][traj_idx]
            asset_ids = dataset_in["asset_ids"][traj_idx]

            args_list.extend(
                [
                    (
                        current_index + img_idx,
                        image_filenames[img_idx].decode("utf-8"),
                        depth_filenames[img_idx].decode("utf-8"),
                        segmentation_filenames[img_idx].decode("utf-8"),
                        pixel_coordinates[img_idx],
                        asset_ids[img_idx],
                        split,
                        lb,
                        ub,
                        output_data_dir,
                    )
                    for img_idx in range(traj_length)
                ]
            )
            current_index += traj_length
            pbar.update(traj_length)

    return args_list


def process_image(args: tuple) -> tuple | None:
    """Process an image and return the new filenames if the segmentation ratio is within the specified bounds."""
    (
        new_img_idx,
        image_filename,
        depth_filename,
        seg_filename,
        pixel_coordinates,
        asset_id,
        split,
        lb,
        ub,
        output_data_dir,
    ) = args
    seg_ratio = calculate_segmentation_ratio(seg_filename, asset_id)

    if lb <= seg_ratio <= ub:
        new_image_filename = f"rgba_{new_img_idx:08d}.png"
        new_depth_filename = f"depth_{new_img_idx:08d}.tiff"
        new_seg_filename = f"segmentation_{new_img_idx:08d}.png"

        # Use split to create the proper folder structure
        split_dir = os.path.join(output_data_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for src_filename, new_filename in zip(
            [image_filename, depth_filename, seg_filename],
            [new_image_filename, new_depth_filename, new_seg_filename],
            strict=False,
        ):
            src_path = os.path.join(ROOT, "data", src_filename)
            dst_path = os.path.join(split_dir, new_filename)
            shutil.copy2(src_path, dst_path)

        local_filename_prefix = output_data_dir.split("data/")[-1] + f"/{split}/"  # get everything after data
        return (
            local_filename_prefix + new_image_filename,
            local_filename_prefix + new_depth_filename,
            local_filename_prefix + new_seg_filename,
            pixel_coordinates,
            asset_id,
        )
    return None  # Explicitly return None if the segmentation ratio doesn't meet criteria


def prune_dataset(
    input_hdf5_path: str, output_hdf5_path: str, output_data_dir: str, lb: float = 0.02, ub: float = 0.5
) -> None:
    """Prune the dataset based on segmentation ratios."""
    print("=" * 80)
    print(f"Pruning dataset with segmentation ratios between {lb} and {ub} and saving to {output_hdf5_path}!")
    print("=" * 80)

    os.makedirs(output_data_dir, exist_ok=True)

    with h5py.File(input_hdf5_path, "r") as f_in, h5py.File(output_hdf5_path, "w") as f_out:
        for split in ["train", "test"]:
            print(f"Pruning {split} dataset...")
            dataset_in = f_in[split]
            dataset_out = f_out.create_group(split)

            args_list = create_args_list(dataset_in, split, lb, ub, output_data_dir)
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(pool.imap(process_image, args_list), total=len(args_list), desc=f"Pruning {split} dataset")
                )

            print("Data pruned! Collating...")
            pruned_data = [r for r in results if r is not None]
            if pruned_data:
                (
                    pruned_image_filenames,
                    pruned_depth_filenames,
                    pruned_segmentation_filenames,
                    pruned_pixel_coordinates,
                    pruned_asset_ids,
                ) = zip(*pruned_data, strict=False)

                # Save pruned data to new HDF5 file
                print("Data collated! Saving pruned HDF5 file...")
                dataset_out.create_dataset("image_filenames", data=pruned_image_filenames)
                dataset_out.create_dataset("depth_filenames", data=pruned_depth_filenames)
                dataset_out.create_dataset("segmentation_filenames", data=pruned_segmentation_filenames)
                dataset_out.create_dataset("pixel_coordinates", data=pruned_pixel_coordinates)
                dataset_out.create_dataset("asset_ids", data=pruned_asset_ids)

        # Copy attributes
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value


if __name__ == "__main__":
    input_hdf5_path = os.path.join(ROOT, "data/merged_lazy/merged.hdf5")
    output_hdf5_path = os.path.join(ROOT, "data/pruned_dataset/pruned.hdf5")
    output_data_dir = os.path.join(ROOT, "data/pruned_dataset/images")
    prune_dataset(input_hdf5_path, output_hdf5_path, output_data_dir)
