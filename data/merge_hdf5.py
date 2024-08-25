import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from perseus import ROOT


def save_images_in_parallel(
    images: list, output_dir: str, mode: str, img_type: str, start_index: int, ext: str = "png"
) -> list:
    """Save images in parallel.

    Args:
        images: list of lists containing images to save.
        output_dir: directory to save the images.
        mode: train or test.
        img_type: images, depth, or segmentation.
        start_index: starting index for the image filenames.
        ext: image extension.

    Returns:
        filenames: list of lists containing saved image filenames, preserving the original shape.
    """
    filenames = []
    image_data = []
    i = start_index

    for traj_idx, img_batch in enumerate(images):
        traj_dir = f"{output_dir}/images/{mode}/{img_type}/traj_{i + traj_idx:08d}"
        os.makedirs(traj_dir, exist_ok=True)

        traj_filenames = []  # To store filenames for this trajectory

        for j, image in enumerate(img_batch):
            if img_type == "image":
                prefix = "rgba"
            elif img_type == "depth":
                prefix = "depth"
            elif img_type == "segmentation":
                prefix = "segmentation"

            save_path = f"{traj_dir}/{prefix}_{j:08d}.{ext}"
            local_path = f"images/{mode}/{img_type}/traj_{i + traj_idx:08d}/{prefix}_{j:08d}.{ext}"
            image_data.append((image, save_path, local_path))
            traj_filenames.append(save_path)  # Collect filenames for this trajectory

        filenames.append(traj_filenames)  # Add trajectory's filenames to the list

    def save_image(image_data: tuple) -> str:
        """Save an image to disk.

        Args:
            image_data: tuple containing the image and the output path.

        Returns:
            local_path: local path to the saved image (to be stored in hdf5).
        """
        image, save_path, _ = image_data
        image = Image.fromarray(image)
        image.save(f"{ROOT}" + save_path)
        return save_path

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(save_image, data): data for data in image_data}
        for future in tqdm(as_completed(futures), desc=f"Saving {mode} {img_type} images", total=len(image_data)):
            future.result()  # Ensures all images are saved

    return filenames


def copy_images_in_parallel(
    src_image_paths: list, output_dir: str, mode: str, img_type: str, start_index: int, ext: str = "png"
) -> list:
    """Copy images in parallel without loading them into memory.

    Args:
        src_image_paths: list of lists containing source image paths to copy.
        output_dir: directory to copy the images to.
        mode: train or test.
        img_type: images, depth, or segmentation.
        start_index: starting index for the image filenames.
        ext: image extension.

    Returns:
        filenames: list of lists containing copied image filenames, preserving the original shape.
    """
    filenames = []
    copy_data = []
    i = start_index

    for traj_idx, img_batch in enumerate(src_image_paths):
        traj_dir = f"{output_dir}/images/{mode}/{img_type}/traj_{i + traj_idx:08d}"
        os.makedirs(traj_dir, exist_ok=True)

        if img_type == "image":
            prefix = "rgba"
        elif img_type == "depth":
            prefix = "depth"
        elif img_type == "segmentation":
            prefix = "segmentation"

        traj_filenames = []

        for j, image_path in enumerate(img_batch):
            save_path = f"{traj_dir}/{prefix}_{j:08d}.{ext}"
            local_path = f"images/{mode}/{img_type}/traj_{i + traj_idx:08d}/{prefix}_{j:08d}.{ext}"
            copy_data.append((image_path, save_path, local_path))
            traj_filenames.append(save_path)

        filenames.append(traj_filenames)

    def copy_image(data: tuple) -> str:
        src_path, dst_path, _ = data
        shutil.copy2(src_path, dst_path)
        return dst_path

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(copy_image, data): data for data in copy_data}
        for future in tqdm(as_completed(futures), desc=f"Copying {mode} {img_type} images", total=len(copy_data)):
            future.result()

    return filenames


def compute_segmentation_ratios(segmentation_paths: list[list[str]]) -> np.ndarray:
    """Compute segmentation ratios using threads.

    Args:
        segmentation_paths: list of lists of segmentation file paths.

    Returns:
        ratios: numpy array of segmentation ratios with shape (num_trajs, num_images_per_traj).
    """

    def compute_ratio(seg_path: str) -> float:
        segmentation = np.array(Image.open(seg_path))
        ratio = np.mean(segmentation > 0)
        return ratio

    num_trajs = len(segmentation_paths)
    num_images_per_traj = len(segmentation_paths[0])
    flattened_paths = [
        (seg, i, j) for i, sub_seg_paths in enumerate(segmentation_paths) for j, seg in enumerate(sub_seg_paths)
    ]

    # computing the segmentation ratios in a parallelized way
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_ratio, seg) for seg, i, j in flattened_paths]
        results = [
            future.result()
            for future in tqdm(as_completed(futures), desc="Computing segmentation ratios", total=len(futures))
        ]

    ratios = np.empty((num_trajs, num_images_per_traj), dtype=float)
    for (_, i, j), ratio in zip(flattened_paths, results, strict=False):
        ratios[i, j] = ratio
    return ratios


def compute_weights(segmentation_ratios: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Compute weights based on the segmentation ratios.

    Args:
        segmentation_ratios: numpy array of segmentation ratios.
        bin_edges: numpy array of bin edges for the histogram.

    Returns:
        weights: numpy array of weights based on the segmentation ratios.
    """
    bin_indices = np.digitize(segmentation_ratios.flatten(), bins=bin_edges, right=True)
    freq = Counter(bin_indices)
    weights = np.zeros(len(bin_indices))
    for bin_idx, count in freq.items():
        weights[bin_indices == bin_idx] = 1.0 / count
    return weights


def merge(  # noqa: PLR0912, PLR0915
    hdf5_list: list, output_dir: str, new_train_frac: float = 0.95, shuffle: bool = False, lazy: bool = True
) -> None:
    """Merge multiple hdf5 datasets into a larger one.

    In particular, also merges all images into a new directory and updates the image filenames.

    Args:
        hdf5_list: list of hdf5 file paths to merge.
        output_dir: directory to save the merged images.
        new_train_frac: fraction of data to use for training after merging (useful for rebalancing datasets).
        shuffle: whether to shuffle the data before splitting.
        lazy: if True, does not save images, depth images, and segmentation images as numpy arrays in the hdf5 file,
            only copies the images in the specified file structure.
    """
    # attributes
    num_keypoints = None
    train_frac = None
    H = None
    W = None

    # all data
    all_train_images = []
    all_train_depth_images = []
    all_train_segmentation_images = []
    all_train_asset_ids = []
    all_train_pixel_coordinates = []
    all_train_object_poses = []
    all_train_object_scales = []
    all_train_camera_poses = []
    all_train_camera_intrinsics = []
    all_orig_train_image_paths = []
    all_orig_train_depth_image_paths = []
    all_orig_train_segmentation_image_paths = []

    all_test_images = []
    all_test_depth_images = []
    all_test_segmentation_images = []
    all_test_asset_ids = []
    all_test_pixel_coordinates = []
    all_test_object_poses = []
    all_test_object_scales = []
    all_test_camera_poses = []
    all_test_camera_intrinsics = []
    all_orig_test_image_paths = []
    all_orig_test_depth_image_paths = []
    all_orig_test_segmentation_image_paths = []

    # aggregating data
    print("Aggregating data...")
    for file_path in hdf5_list:
        with h5py.File(file_path, "r") as f:
            # attributes
            if num_keypoints is None:
                num_keypoints = f.attrs["num_keypoints"]

            if train_frac is None:
                train_frac = f.attrs["train_frac"]

            if H is None:
                H = f.attrs["H"]

            if W is None:
                W = f.attrs["W"]

            # train data
            if not lazy:
                all_train_images.append(f["train"]["images"][()])
                all_train_depth_images.append(f["train"]["depth_images"][()])
                all_train_segmentation_images.append(f["train"]["segmentation_images"][()])
            all_train_asset_ids.append(f["train"]["asset_ids"][()])
            all_train_pixel_coordinates.append(f["train"]["pixel_coordinates"][()])
            all_train_object_poses.append(f["train"]["object_poses"][()])
            all_train_object_scales.append(f["train"]["object_scales"][()])
            all_train_camera_poses.append(f["train"]["camera_poses"][()])
            all_train_camera_intrinsics.append(f["train"]["camera_intrinsics"][()])

            all_orig_train_image_paths.append(f["train"]["image_filenames"][()])
            all_orig_train_depth_image_paths.append(f["train"]["depth_filenames"][()])
            all_orig_train_segmentation_image_paths.append(f["train"]["segmentation_filenames"][()])

            # test data
            if not lazy:
                all_test_images.append(f["test"]["images"][()])
                all_test_depth_images.append(f["test"]["depth_images"][()])
                all_test_segmentation_images.append(f["test"]["segmentation_images"][()])
            all_test_asset_ids.append(f["test"]["asset_ids"][()])
            all_test_pixel_coordinates.append(f["test"]["pixel_coordinates"][()])
            all_test_object_poses.append(f["test"]["object_poses"][()])
            all_test_object_scales.append(f["test"]["object_scales"][()])
            all_test_camera_poses.append(f["test"]["camera_poses"][()])
            all_test_camera_intrinsics.append(f["test"]["camera_intrinsics"][()])

            all_orig_test_image_paths.append(f["test"]["image_filenames"][()])
            all_orig_test_depth_image_paths.append(f["test"]["depth_filenames"][()])
            all_orig_test_segmentation_image_paths.append(f["test"]["segmentation_filenames"][()])

    # converting to numpy
    print("Converting to numpy...")
    if not lazy:
        all_train_images = np.concatenate(all_train_images, axis=0)
        all_train_depth_images = np.concatenate(all_train_depth_images, axis=0)
        all_train_segmentation_images = np.concatenate(all_train_segmentation_images, axis=0)
    all_train_asset_ids = np.concatenate(all_train_asset_ids, axis=0)
    all_train_pixel_coordinates = np.concatenate(all_train_pixel_coordinates, axis=0)
    all_train_object_poses = np.concatenate(all_train_object_poses, axis=0)
    all_train_object_scales = np.concatenate(all_train_object_scales, axis=0)
    all_train_camera_poses = np.concatenate(all_train_camera_poses, axis=0)
    all_train_camera_intrinsics = np.concatenate(all_train_camera_intrinsics, axis=0)

    all_train_image_paths = np.concatenate(all_orig_train_image_paths, axis=0)
    all_train_depth_image_paths = np.concatenate(all_orig_train_depth_image_paths, axis=0)
    all_train_segmentation_image_paths = np.concatenate(all_orig_train_segmentation_image_paths, axis=0)

    if not lazy:
        all_test_images = np.concatenate(all_test_images, axis=0)
        all_test_depth_images = np.concatenate(all_test_depth_images, axis=0)
        all_test_segmentation_images = np.concatenate(all_test_segmentation_images, axis=0)
    all_test_asset_ids = np.concatenate(all_test_asset_ids, axis=0)
    all_test_pixel_coordinates = np.concatenate(all_test_pixel_coordinates, axis=0)
    all_test_object_poses = np.concatenate(all_test_object_poses, axis=0)
    all_test_object_scales = np.concatenate(all_test_object_scales, axis=0)
    all_test_camera_poses = np.concatenate(all_test_camera_poses, axis=0)
    all_test_camera_intrinsics = np.concatenate(all_test_camera_intrinsics, axis=0)

    all_test_image_paths = np.concatenate(all_orig_test_image_paths, axis=0)
    all_test_depth_image_paths = np.concatenate(all_orig_test_depth_image_paths, axis=0)
    all_test_segmentation_image_paths = np.concatenate(all_orig_test_segmentation_image_paths, axis=0)

    # shuffling new dataset if requested
    if shuffle:
        print("Shuffling data...")
        num_train = int(new_train_frac * (len(all_train_images) + len(all_test_images)))
        if not lazy:
            all_images = np.concatenate([all_train_images, all_test_images], axis=0)
            all_depth_images = np.concatenate([all_train_depth_images, all_test_depth_images], axis=0)
            all_segmentation_images = np.concatenate(
                [all_train_segmentation_images, all_test_segmentation_images], axis=0
            )
        all_asset_ids = np.concatenate([all_train_asset_ids, all_test_asset_ids], axis=0)
        all_pixel_coordinates = np.concatenate([all_train_pixel_coordinates, all_test_pixel_coordinates], axis=0)
        all_object_poses = np.concatenate([all_train_object_poses, all_test_object_poses], axis=0)
        all_object_scales = np.concatenate([all_train_object_scales, all_test_object_scales], axis=0)
        all_camera_poses = np.concatenate([all_train_camera_poses, all_test_camera_poses], axis=0)
        all_camera_intrinsics = np.concatenate([all_train_camera_intrinsics, all_test_camera_intrinsics], axis=0)

        all_image_paths = np.concatenate([all_train_image_paths, all_test_image_paths], axis=0)
        all_depth_image_paths = np.concatenate([all_train_depth_image_paths, all_test_depth_image_paths], axis=0)
        all_segmentation_image_paths = np.concatenate(
            [all_train_segmentation_image_paths, all_test_segmentation_image_paths], axis=0
        )

        indices = np.random.permutation(len(all_images))
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]

        if not lazy:
            train_images = all_images[train_indices]
            train_depth_images = all_depth_images[train_indices]
            train_segmentation_images = all_segmentation_images[train_indices]
        train_asset_ids = all_asset_ids[train_indices]
        train_pixel_coordinates = all_pixel_coordinates[train_indices]
        train_object_poses = all_object_poses[train_indices]
        train_object_scales = all_object_scales[train_indices]
        train_camera_poses = all_camera_poses[train_indices]
        train_camera_intrinsics = all_camera_intrinsics[train_indices]

        train_image_paths = all_image_paths[train_indices]
        train_depth_image_paths = all_depth_image_paths[train_indices]
        train_segmentation_image_paths = all_segmentation_image_paths[train_indices]

        if not lazy:
            test_images = all_images[test_indices]
            test_depth_images = all_depth_images[test_indices]
            test_segmentation_images = all_segmentation_images[test_indices]
        test_asset_ids = all_asset_ids[test_indices]
        test_pixel_coordinates = all_pixel_coordinates[test_indices]
        test_object_poses = all_object_poses[test_indices]
        test_object_scales = all_object_scales[test_indices]
        test_camera_poses = all_camera_poses[test_indices]
        test_camera_intrinsics = all_camera_intrinsics[test_indices]

        test_image_paths = all_image_paths[test_indices]
        test_depth_image_paths = all_depth_image_paths[test_indices]
        test_segmentation_image_paths = all_segmentation_image_paths[test_indices]

    else:
        print("Merging data...")
        if not lazy:
            train_images = all_train_images
            train_depth_images = all_train_depth_images
            train_segmentation_images = all_train_segmentation_images
        train_asset_ids = all_train_asset_ids
        train_pixel_coordinates = all_train_pixel_coordinates
        train_object_poses = all_train_object_poses
        train_object_scales = all_train_object_scales
        train_camera_poses = all_train_camera_poses
        train_camera_intrinsics = all_train_camera_intrinsics

        train_image_paths = all_train_image_paths
        train_depth_image_paths = all_train_depth_image_paths
        train_segmentation_image_paths = all_train_segmentation_image_paths

        if not lazy:
            test_images = all_test_images
            test_depth_images = all_test_depth_images
            test_segmentation_images = all_test_segmentation_images
        test_asset_ids = all_test_asset_ids
        test_pixel_coordinates = all_test_pixel_coordinates
        test_object_poses = all_test_object_poses
        test_object_scales = all_test_object_scales
        test_camera_poses = all_test_camera_poses
        test_camera_intrinsics = all_test_camera_intrinsics

        test_image_paths = all_test_image_paths
        test_depth_image_paths = all_test_depth_image_paths
        test_segmentation_image_paths = all_test_segmentation_image_paths

    # resaving images under a new directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(f"{output_dir}/train")
        os.makedirs(f"{output_dir}/test")
    else:
        raise ValueError(
            f"Directory {output_dir} already exists! For safety, please manually remove it or choose a new directory."
        )  # TODO(ahl): deal with this in a smarter way

    if lazy:
        train_image_filenames = copy_images_in_parallel(
            train_image_paths, output_dir.split("/")[-1], "train", "image", 0
        )
        test_image_filenames = copy_images_in_parallel(test_image_paths, output_dir.split("/")[-1], "test", "image", 0)
        train_depth_filenames = copy_images_in_parallel(
            train_depth_image_paths, output_dir.split("/")[-1], "train", "depth", 0, ext="tiff"
        )
        test_depth_filenames = copy_images_in_parallel(
            test_depth_image_paths, output_dir.split("/")[-1], "test", "depth", 0, ext="tiff"
        )
        train_segmentation_filenames = copy_images_in_parallel(
            train_segmentation_image_paths, output_dir.split("/")[-1], "train", "segmentation", 0
        )
        test_segmentation_filenames = copy_images_in_parallel(
            test_segmentation_image_paths, output_dir.split("/")[-1], "test", "segmentation", 0
        )
    else:
        train_image_filenames = save_images_in_parallel(train_images, output_dir.split("/")[-1], "train", "image", 0)
        test_image_filenames = save_images_in_parallel(test_images, output_dir.split("/")[-1], "test", "image", 0)
        train_depth_filenames = save_images_in_parallel(
            train_depth_images, output_dir.split("/")[-1], "train", "depth", 0, ext="tiff"
        )
        test_depth_filenames = save_images_in_parallel(
            test_depth_images, output_dir.split("/")[-1], "test", "depth", 0, ext="tiff"
        )
        train_segmentation_filenames = save_images_in_parallel(
            train_segmentation_images, output_dir.split("/")[-1], "train", "segmentation", 0
        )
        test_segmentation_filenames = save_images_in_parallel(
            test_segmentation_images, output_dir.split("/")[-1], "test", "segmentation", 0
        )

    # computing segmentation ratios
    train_segmentation_ratios = compute_segmentation_ratios(train_segmentation_image_paths)
    test_segmentation_ratios = compute_segmentation_ratios(test_segmentation_image_paths)

    # computing frequency-based weights for each of the datapoints
    bin_edges = np.linspace(0, 1, 100)  # 100 bins for segmentation ratios
    train_weights = compute_weights(train_segmentation_ratios, bin_edges)
    test_weights = compute_weights(test_segmentation_ratios, bin_edges)

    # creating new dataset
    print("Creating new dataset...")
    with h5py.File(f"{output_dir}/merged.hdf5", "w") as f:
        # attributes
        f.attrs["num_keypoints"] = num_keypoints
        f.attrs["train_frac"] = train_frac
        f.attrs["H"] = H
        f.attrs["W"] = W

        # train data
        train = f.create_group("train")
        if not lazy:
            train.create_dataset("images", data=train_images)
            train.create_dataset("depth_images", data=train_depth_images)
            train.create_dataset("segmentation_images", data=train_segmentation_images)
        train.create_dataset("asset_ids", data=train_asset_ids)
        train.create_dataset("pixel_coordinates", data=train_pixel_coordinates)
        train.create_dataset("object_poses", data=train_object_poses)
        train.create_dataset("object_scales", data=train_object_scales)
        train.create_dataset("camera_poses", data=train_camera_poses)
        train.create_dataset("camera_intrinsics", data=train_camera_intrinsics)
        train.create_dataset("image_filenames", data=train_image_filenames)
        train.create_dataset("depth_filenames", data=train_depth_filenames)
        train.create_dataset("segmentation_filenames", data=train_segmentation_filenames)
        train.create_dataset("segmentation_ratios", data=train_segmentation_ratios)
        train.create_dataset("weights", data=train_weights)

        # test data
        test = f.create_group("test")
        if not lazy:
            test.create_dataset("images", data=test_images)
            test.create_dataset("depth_images", data=test_depth_images)
            test.create_dataset("segmentation_images", data=test_segmentation_images)
        test.create_dataset("asset_ids", data=test_asset_ids)
        test.create_dataset("pixel_coordinates", data=test_pixel_coordinates)
        test.create_dataset("object_poses", data=test_object_poses)
        test.create_dataset("object_scales", data=test_object_scales)
        test.create_dataset("camera_poses", data=test_camera_poses)
        test.create_dataset("camera_intrinsics", data=test_camera_intrinsics)
        test.create_dataset("image_filenames", data=test_image_filenames)
        test.create_dataset("depth_filenames", data=test_depth_filenames)
        test.create_dataset("segmentation_filenames", data=test_segmentation_filenames)
        test.create_dataset("segmentation_ratios", data=test_segmentation_ratios)
        test.create_dataset("weights", data=test_weights)


if __name__ == "__main__":
    hdf5_list = [
        f"{ROOT}/data/qwerty1/mjc_data.hdf5",
        f"{ROOT}/data/qwerty2/mjc_data.hdf5",
        f"{ROOT}/data/qwerty3/mjc_data.hdf5",
        f"{ROOT}/data/qwerty4/mjc_data.hdf5",
        f"{ROOT}/data/qwerty5/mjc_data.hdf5",
        f"{ROOT}/data/qwerty6/mjc_data.hdf5",
        f"{ROOT}/data/qwerty7/mjc_data.hdf5",
        f"{ROOT}/data/qwerty8/mjc_data.hdf5",
        f"{ROOT}/data/qwerty9/mjc_data.hdf5",
        f"{ROOT}/data/qwerty10/mjc_data.hdf5",
        f"{ROOT}/data/qwerty11/mjc_data.hdf5",
        f"{ROOT}/data/qwerty12/mjc_data.hdf5",
        f"{ROOT}/data/qwerty13/mjc_data.hdf5",
        f"{ROOT}/data/qwerty14/mjc_data.hdf5",
        f"{ROOT}/data/qwerty15/mjc_data.hdf5",
        f"{ROOT}/data/qwerty16/mjc_data.hdf5",
        f"{ROOT}/data/qwerty17/mjc_data.hdf5",
    ]
    output_dir = f"{ROOT}/data/merged_lazy"
    merge(hdf5_list, output_dir, shuffle=False)
    print("Merging complete.")
