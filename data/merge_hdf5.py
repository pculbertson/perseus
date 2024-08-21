import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from perseus import ROOT


def save_image(image_data: tuple) -> str:
    """Save an image to disk.

    Args:
        image_data: tuple containing the image and the output path.

    Returns:
        output_path: path to the saved image.
    """
    image, output_path = image_data
    image = Image.fromarray(image)
    image.save(output_path)
    return output_path


def save_images_in_parallel(images: list, output_dir: str, mode: str, start_index: int) -> list:
    """Save images in parallel.

    Args:
        images: list of images to save.
        output_dir: directory to save the images.
        mode: train or test.
        start_index: starting index for the image filenames.

    Returns:
        filenames: list of saved image filenames.
    """
    filenames = []
    image_data = []
    i = start_index

    for img_batch in images:
        for image in img_batch:
            output_path = f"{output_dir}/images/{mode}/img{i:08d}.png"
            image_data.append((image, output_path))
            i += 1

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(save_image, data): data for data in image_data}
        for future in tqdm(as_completed(futures), desc=f"Saving {mode} images", total=len(image_data)):
            filenames.append(future.result())

    return filenames


def merge(hdf5_list: list, output_dir: str) -> None:  # noqa: PLR0915
    """Merge multiple hdf5 datasets into a larger one.

    In particular, also merges all images into a new directory and updates the image filenames.

    Args:
        hdf5_list: list of hdf5 file paths to merge.
        output_dir: directory to save the merged images.
    """
    # attributes
    num_keypoints = None
    train_frac = None
    H = None
    W = None

    # train data
    train_images = []
    train_pixel_coordinates = []
    train_object_poses = []
    train_object_scales = []
    train_camera_poses = []
    train_camera_intrinsics = []
    train_image_filenames = []

    # test data
    test_images = []
    test_pixel_coordinates = []
    test_object_poses = []
    test_object_scales = []
    test_camera_poses = []
    test_camera_intrinsics = []
    test_image_filenames = []

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
            train_images.append(f["train"]["images"][()])
            train_pixel_coordinates.append(f["train"]["pixel_coordinates"][()])
            train_object_poses.append(f["train"]["object_poses"][()])
            train_object_scales.append(f["train"]["object_scales"][()])
            train_camera_poses.append(f["train"]["camera_poses"][()])
            train_camera_intrinsics.append(f["train"]["camera_intrinsics"][()])

            # test data
            test_images.append(f["test"]["images"][()])
            test_pixel_coordinates.append(f["test"]["pixel_coordinates"][()])
            test_object_poses.append(f["test"]["object_poses"][()])
            test_object_scales.append(f["test"]["object_scales"][()])
            test_camera_poses.append(f["test"]["camera_poses"][()])
            test_camera_intrinsics.append(f["test"]["camera_intrinsics"][()])

    # converting to numpy
    print("Converting to numpy...")
    train_images = np.concatenate(train_images, axis=0)
    train_pixel_coordinates = np.concatenate(train_pixel_coordinates, axis=0)
    train_object_poses = np.concatenate(train_object_poses, axis=0)
    train_object_scales = np.concatenate(train_object_scales, axis=0)
    train_camera_poses = np.concatenate(train_camera_poses, axis=0)
    train_camera_intrinsics = np.concatenate(train_camera_intrinsics, axis=0)
    train_image_filenames = list(itertools.chain(*train_image_filenames))

    test_images = np.concatenate(test_images, axis=0)
    test_pixel_coordinates = np.concatenate(test_pixel_coordinates, axis=0)
    test_object_poses = np.concatenate(test_object_poses, axis=0)
    test_object_scales = np.concatenate(test_object_scales, axis=0)
    test_camera_poses = np.concatenate(test_camera_poses, axis=0)
    test_camera_intrinsics = np.concatenate(test_camera_intrinsics, axis=0)
    test_image_filenames = list(itertools.chain(*test_image_filenames))

    # resaving images under a new directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(f"{output_dir}/images/train")
        os.makedirs(f"{output_dir}/images/test")
    else:
        raise ValueError(
            f"Directory {output_dir} already exists! For safety, please manually remove it or choose a new directory."
        )  # TODO(ahl): deal with this in a smarter way

    train_image_filenames = save_images_in_parallel(train_images, output_dir.split("/")[-1], "train", 0)
    test_image_filenames = save_images_in_parallel(test_images, output_dir.split("/")[-1], "test", 0)

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
        train.create_dataset("images", data=train_images)
        train.create_dataset("pixel_coordinates", data=train_pixel_coordinates)
        train.create_dataset("object_poses", data=train_object_poses)
        train.create_dataset("object_scales", data=train_object_scales)
        train.create_dataset("camera_poses", data=train_camera_poses)
        train.create_dataset("camera_intrinsics", data=train_camera_intrinsics)
        train.create_dataset("image_filenames", data=train_image_filenames)

        # test data
        test = f.create_group("test")
        test.create_dataset("images", data=test_images)
        test.create_dataset("pixel_coordinates", data=test_pixel_coordinates)
        test.create_dataset("object_poses", data=test_object_poses)
        test.create_dataset("object_scales", data=test_object_scales)
        test.create_dataset("camera_poses", data=test_camera_poses)
        test.create_dataset("camera_intrinsics", data=test_camera_intrinsics)
        test.create_dataset("image_filenames", data=test_image_filenames)


if __name__ == "__main__":
    hdf5_list = [
        f"{ROOT}/data/qwerty_aggregated/mjc_data.hdf5",
        f"{ROOT}/data/qwerty_aggregated2/mjc_data.hdf5",
        f"{ROOT}/data/qwerty_aggregated3/mjc_data.hdf5",
    ]
    output_dir = f"{ROOT}/data/merged"
    merge(hdf5_list, output_dir)
    print("Merging complete.")
