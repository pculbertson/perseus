import numpy as np
import pypose as pp
import torch


def to_SE3(positions, quaternions):
    p = torch.as_tensor(positions)
    q = torch.as_tensor(quaternions)

    return pp.SE3(torch.cat([p, q], dim=-1))


def get_pixel_coordinates(
    keypoints: torch.Tensor,
    object_poses: pp.SE3_type,
    camera_poses: pp.SE3_type,
    camera_matrix: torch.Tensor,
):
    """Get pixel coordinates of keypoints.

    Args:
        keypoints: (n_points, 3) tensor of keypoints in world coordinates.
        object_poses: (n_frames, 7) SE3 tensor of object poses.
        camera_poses: (n_frames, 7) SE3 tensor of camera poses.
        camera_matrix: (3, 3) tensor of camera matrix."""
    # Convert keypoints to world coordinates.
    print(keypoints.shape, object_poses.shape, camera_poses.shape, camera_matrix.shape)
    keypoints_world = object_poses.unsqueeze(1).Inv().Act(keypoints.unsqueeze(0))

    # Project points to camera coordinates.
    return pp.point2pixel(
        keypoints_world,
        camera_matrix.unsqueeze(0).unsqueeze(0),
        extrinsics=camera_poses.unsqueeze(1),
    )
