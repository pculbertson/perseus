import json
from kubric import ArgumentParser
import trimesh
import os
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm
from data_generation.data_utils import to_SE3, get_pixel_coordinates, reorder_quaternion
import h5py
import numpy as np
import copy
from PIL import Image
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "--job-id", type=str, help="ID of job to process (for debug). Default: None."
)
parser.add_argument(
    "--asset_id", type=str, help="ID of asset to process. Default: mjc.", default="mjc"
)
parser.add_argument(
    "--num_keypoints",
    type=int,
    help="Number of keypoints to generate. Default: 32.",
    default=32,
)

parser.add_argument(
    "--train-frac",
    type=float,
    help="Fraction of data to use for training. Default: 0.8.",
    default=0.8,
)

parser.add_argument(
    "--debug-plot",
    action="store_true",
    help="Plot debug images. Default: False.",
    default=False,
)


def get_keypoints(args):
    """Get keypoints for a given job."""
    # Load keypoints.
    keypoints_filename = os.path.join(args.job_dir, f"{args.asset_id}_keypoints.json")
    if os.path.exists(keypoints_filename):
        with open(keypoints_filename, "r") as f:
            keypoints = json.load(f)
    else:
        keypoints = generate_keypoints(args)

    # Return keypoints.
    return keypoints


def generate_keypoints(args):
    """Generate keypoints for a given model."""
    # Load object.
    metadata_filename = os.path.join(args.job_dir, args.job_id, "metadata.json")
    with open(metadata_filename, "r") as f:
        metadata = json.load(f)

    # Get model filename.
    asset_info = [
        dd for dd in metadata["instances"] if dd["asset_id"] == args.asset_id
    ][0]
    model_filename = asset_info["render_filename"]

    mesh = trimesh.load(model_filename, force="mesh")

    # Generate keypoints.
    keypoints = trimesh.sample.sample_surface(mesh, args.num_keypoints)[0]

    # Save keypoints.
    keypoints_filename = os.path.join(args.job_dir, f"{args.asset_id}_keypoints.json")
    with open(keypoints_filename, "w") as f:
        json.dump(keypoints.tolist(), f)

    # Return keypoints.
    return keypoints


def generate_data(args):
    """Generate data for a given job."""

    # Load job metadata.
    metadata_filename = os.path.join(args.job_dir, args.job_id, "metadata.json")
    with open(metadata_filename, "r") as f:
        metadata = json.load(f)

    # Get camera intrinsics and extrinsics.
    fov = metadata["camera"]["field_of_view"]
    H = metadata["flags"]["resolution"]
    W = metadata["flags"]["resolution"]
    camera_positions = torch.as_tensor(metadata["camera"]["positions"])
    camera_quaternions = reorder_quaternion(
        torch.as_tensor(metadata["camera"]["quaternions"])
    )
    camera_poses = to_SE3(camera_positions, camera_quaternions)

    # Get object poses.
    object_dict = [
        dd for dd in metadata["instances"] if dd["asset_id"] == args.asset_id
    ][0]
    object_abs_scale = torch.as_tensor(object_dict["abs_scale"])
    object_positions = torch.as_tensor(object_dict["positions"])
    object_quaternions = reorder_quaternion(torch.as_tensor(object_dict["quaternions"]))
    object_poses = to_SE3(object_positions, object_quaternions)

    # Get keypoints.
    keypoints = torch.as_tensor(args.keypoints).float() * object_abs_scale

    # Get pixel coordinates.
    pixel_coordinates = get_pixel_coordinates(
        keypoints, object_poses, camera_poses, fov, H, W
    )

    # Get object scale.
    object_scales = (
        torch.as_tensor(object_dict["abs_scale"]).float().unsqueeze(0).expand(24)
    )

    # Compute camera intrinsics.
    f_x = W / (2 * np.tan(fov / 2))
    f_y = H / (2 * np.tan(fov / 2))
    camera_intrinsics = torch.tensor(
        [[f_x, 0, W / 2], [0, f_y, H / 2], [0, 0, 1]], dtype=torch.float32
    )

    camera_intrinsics = camera_intrinsics.unsqueeze(0).expand(24, -1, -1)

    # Load all rgb images in job dir.
    rgb_filenames = [
        os.path.join(args.job_dir, args.job_id, f"rgba_{ii:05d}.png")
        for ii in range(24)
    ]

    # Create an empty list to store the images
    rgb_images = []

    # Iterate over each RGB image filename
    for filename in rgb_filenames:
        # Read the image using OpenCV
        image = Image.open(filename).convert("RGB")

        # Append the image to the list
        rgb_images.append(np.array(image))

    return (
        rgb_images,
        pixel_coordinates.cpu().numpy(),
        object_poses.data.cpu().numpy(),
        object_scales,
        camera_poses,
        camera_intrinsics,
        rgb_filenames,
    )


def plot_data(args):
    # Debug plotting to check that everything is working.
    import matplotlib.pyplot as plt

    # Load pixel coordinates.
    pixel_coordinates_filename = os.path.join(
        args.job_dir, args.job_id, f"{args.asset_id}_pixel_coordinates.json"
    )

    with open(pixel_coordinates_filename, "r") as f:
        pixel_coordinates = json.load(f)

    # Load all rgb images in job dir.
    rgb_filenames = [
        os.path.join(args.job_dir, args.job_id, f"rgba_{ii:05d}.png")
        for ii in range(24)
    ]

    # Create debug directory if it doesn't exist.
    if not os.path.exists(os.path.join(args.job_dir, "debug")):
        os.makedirs(os.path.join(args.job_dir, "debug"))

    # Plot pixel coordinates on top of rgb images.
    for ii, (rgb_filename, pixel_coordinate) in enumerate(
        zip(rgb_filenames, pixel_coordinates)
    ):
        rgb = plt.imread(rgb_filename)
        pixel_coordinate = np.array(pixel_coordinate)

        fig, ax = plt.subplots()
        ax.imshow(rgb, aspect="auto")
        plt.axis("off")

        # Scatter points with different color for each keypoint.
        for jj in range(args.num_keypoints):
            plt.scatter(
                pixel_coordinate[jj, 0],
                pixel_coordinate[jj, 1],
                c=jj,
                s=10,
            )

        plt.savefig(
            os.path.join(args.job_dir, "debug", f"debug_keypoints_{ii}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


def main(args):
    job_ids = (
        [args.job_id]
        if args.job_id
        else [
            ff
            for ff in os.listdir(args.job_dir)
            if os.path.isdir(os.path.join(args.job_dir, ff))
        ]
    )

    args_list = [copy.deepcopy(args) for _ in range(len(job_ids))]
    for jj, aa in zip(job_ids, args_list):
        aa.job_id = jj

    keypoints = get_keypoints(args_list[0])
    for aa in args_list:
        aa.keypoints = keypoints

    for aa in args_list:
        print(aa.job_id)

    image_list = []
    pixel_coords_list = []
    obj_poses_list = []
    obj_scales_list = []
    camera_poses_list = []
    camera_intrinsics_list = []
    image_filename_list = []

    # Wrap for loop in tqdm
    for aa in tqdm(args_list):
        try:
            (
                images,
                pixel_coords,
                obj_poses,
                obj_scales,
                camera_poses,
                camera_intrinsics,
                image_filenames,
            ) = generate_data(aa)
        except Exception as e:
            print(f"Failed to generate data for job {aa.job_id}.")
            print(e)
            continue
        image_list.append(np.stack(images))
        pixel_coords_list.append(np.stack(pixel_coords))
        obj_poses_list.append(np.stack(obj_poses))
        obj_scales_list.append(np.stack(obj_scales))
        camera_poses_list.append(np.stack(camera_poses))
        camera_intrinsics_list.append(np.stack(camera_intrinsics))
        image_filename_list.append(np.stack(image_filenames))

    # Concatenate data and cast to torch.
    image_list = np.stack(image_list, axis=0)
    pixel_coords_list = np.stack(pixel_coords_list, axis=0)
    obj_poses_list = np.stack(obj_poses_list, axis=0)
    obj_scales_list = np.stack(obj_scales_list, axis=0)
    camera_poses_list = np.stack(camera_poses_list, axis=0)
    camera_intrinsics_list = np.stack(camera_intrinsics_list, axis=0)
    image_filename_list = np.stack(image_filename_list, axis=0).astype("S")

    # Save data as hdf5 file.
    split_idx = int(image_list.shape[0] * args.train_frac)

    data_filename = os.path.join(args.job_dir, f"{args.asset_id}_data.hdf5")
    with h5py.File(data_filename, "w") as f:
        # Store training data.
        train = f.create_group("train")
        train.create_dataset("images", data=torch.from_numpy(image_list[:split_idx]))
        train.create_dataset(
            "pixel_coordinates",
            data=torch.from_numpy(pixel_coords_list[:split_idx]),
        )
        train.create_dataset(
            "object_poses", data=torch.from_numpy(obj_poses_list[:split_idx])
        )
        train.create_dataset(
            "object_scales", data=torch.from_numpy(obj_scales_list[:split_idx])
        )
        train.create_dataset(
            "camera_poses", data=torch.from_numpy(camera_poses_list[:split_idx])
        )
        train.create_dataset(
            "camera_intrinsics",
            data=torch.from_numpy(camera_intrinsics_list[:split_idx]),
        )
        train.create_dataset("image_filenames", data=image_filename_list[:split_idx])

        # Store test data.
        test = f.create_group("test")
        test.create_dataset("images", data=torch.from_numpy(image_list[split_idx:]))
        test.create_dataset(
            "pixel_coordinates",
            data=torch.from_numpy(pixel_coords_list[split_idx:]),
        )
        test.create_dataset(
            "object_poses", data=torch.from_numpy(obj_poses_list[split_idx:])
        )
        test.create_dataset(
            "object_scales", data=torch.from_numpy(obj_scales_list[split_idx:])
        )
        test.create_dataset(
            "camera_poses", data=torch.from_numpy(camera_poses_list[split_idx:])
        )
        test.create_dataset(
            "camera_intrinsics",
            data=torch.from_numpy(camera_intrinsics_list[split_idx:]),
        )
        test.create_dataset("image_filenames", data=image_filename_list[split_idx:])

        # Store hyperparameters.
        f.attrs["num_keypoints"] = args.num_keypoints
        f.attrs["train_frac"] = args.train_frac
        f.attrs["H"] = image_list.shape[1]
        f.attrs["W"] = image_list.shape[2]

    if args.debug_plot:
        plot_data(args_list[0])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
