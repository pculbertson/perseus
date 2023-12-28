import json
from kubric import ArgumentParser
import trimesh
import os
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm
from data_generation.data_utils import to_SE3, get_pixel_coordinates, reorder_quaternion

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
    print(object_dict.keys())
    print(metadata["instances"])
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

    # Save pixel coordinates.
    pixel_coordinates_filename = os.path.join(
        args.job_dir, args.job_id, f"{args.asset_id}_pixel_coordinates.json"
    )

    with open(pixel_coordinates_filename, "w") as f:
        json.dump(pixel_coordinates.tolist(), f)


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

    args_list = [args for _ in range(len(job_ids))]
    for jj, aa in zip(job_ids, args_list):
        aa.job_id = jj

    keypoints = get_keypoints(args_list[0])
    for aa in args_list:
        aa.keypoints = keypoints

    for aa in args_list:
        generate_data(aa)

    if args.debug_plot:
        plot_data(args_list[-1])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
