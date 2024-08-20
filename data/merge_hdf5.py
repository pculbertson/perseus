import itertools

import h5py
import numpy as np

from perseus import ROOT


def merge(hdf5_list: list) -> None:  # noqa: PLR0915
    """Merge multiple hdf5 datasets into a larger one.

    Args:
        hdf5_list: list of hdf5 file paths to merge.
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

    # looping through hdf5 files
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
            train_images.append(f["train"]["images"][:])
            train_pixel_coordinates.append(f["train"]["pixel_coordinates"][:])
            train_object_poses.append(f["train"]["object_poses"][:])
            train_object_scales.append(f["train"]["object_scales"][:])
            train_camera_poses.append(f["train"]["camera_poses"][:])
            train_camera_intrinsics.append(f["train"]["camera_intrinsics"][:])
            train_image_filenames.append(f["train"]["image_filenames"][:])

            # test data
            test_images.append(f["test"]["images"][:])
            test_pixel_coordinates.append(f["test"]["pixel_coordinates"][:])
            test_object_poses.append(f["test"]["object_poses"][:])
            test_object_scales.append(f["test"]["object_scales"][:])
            test_camera_poses.append(f["test"]["camera_poses"][:])
            test_camera_intrinsics.append(f["test"]["camera_intrinsics"][:])
            test_image_filenames.append(f["test"]["image_filenames"][:])

    # converting to numpy
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

    # creating new dataset
    with h5py.File("merged.hdf5", "w") as f:
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
    merge(hdf5_list)
    print("Merging complete.")