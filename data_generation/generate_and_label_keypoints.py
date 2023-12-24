import json
from kubric import ArgumentParser
import trimesh
import os
import numpy as np
import multiprocessing
import torch
from tqdm import tqdm
from data_generation.data_utils import to_SE3, get_pixel_coordinates

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
    camera_matrix = torch.as_tensor(metadata["camera"]["K"])
    camera_positions = torch.as_tensor(metadata["camera"]["positions"])
    camera_quaternions = torch.as_tensor(metadata["camera"]["quaternions"])
    camera_poses = to_SE3(camera_positions, camera_quaternions)

    # Get object poses.
    object_dict = [
        dd for dd in metadata["instances"] if dd["asset_id"] == args.asset_id
    ][0]
    object_positions = torch.as_tensor(object_dict["positions"])
    object_quaternions = torch.as_tensor(object_dict["quaternions"])
    object_poses = to_SE3(object_positions, object_quaternions)

    # Get keypoints.
    keypoints = torch.as_tensor(args.keypoints)

    # Get pixel coordinates.
    pixel_coordinates = get_pixel_coordinates(
        keypoints, object_poses, camera_poses, camera_matrix
    )

    # Save pixel coordinates.
    pixel_coordinates_filename = os.path.join(
        args.job_dir, args.job_id, f"{args.asset_id}_pixel_coordinates.json"
    )

    with open(pixel_coordinates_filename, "w") as f:
        json.dump(pixel_coordinates.tolist(), f)


def main(args):
    job_ids = [args.job_id] if args.job_id else os.listdir(args.job_dir)

    args_list = [args for _ in range(len(job_ids))]
    for jj, aa in zip(job_ids, args_list):
        aa.job_id = jj

    keypoints = get_keypoints(args_list[0])
    for aa in args_list:
        aa.keypoints = keypoints

    for aa in args_list:
        generate_data(aa)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
