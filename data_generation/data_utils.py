import numpy as np
import pypose as pp
import torch


def to_SE3(positions, quaternions):
    p = torch.as_tensor(positions)
    q = torch.as_tensor(quaternions)

    return pp.SE3(torch.cat([p, q], dim=-1))


def reorder_quaternion(quaternions):
    return torch.cat([quaternions[..., 1:], quaternions[..., :1]], dim=-1)


def get_pixel_coordinates(
    keypoints: torch.Tensor,
    object_poses: pp.SE3_type,
    camera_poses: pp.SE3_type,
    fov: float,
    H: int,
    W: int,
):
    """Get pixel coordinates of keypoints.

    Args:
        keypoints: (n_points, 3) tensor of keypoints in world coordinates.
        object_poses: (n_frames, 7) SE3 tensor of object poses.
        camera_poses: (n_frames, 7) SE3 tensor of camera poses.
        fov: Field of view of camera.
        H: Image height.
        W: Image width.
    """

    n_frames, _ = camera_poses.shape

    # Compute conversion from Blender to OpenCV.
    blender_to_opencv = (
        pp.euler2SO3(torch.tensor([np.pi, 0, 0])).unsqueeze(0).expand(n_frames, -1)
    )

    blender_to_opencv = pp.SE3(
        torch.cat([torch.zeros(n_frames, 3), blender_to_opencv], dim=-1)
    )

    camera_poses = camera_poses @ blender_to_opencv

    # Convert keypoints to world coordinates.
    camera_to_object = camera_poses.Inv() @ object_poses

    # Compute intrinsics matrix.
    f_x = W / (2 * np.tan(fov / 2))
    f_y = H / (2 * np.tan(fov / 2))
    camera_matrix = torch.tensor(
        [[f_x, 0, W / 2], [0, f_y, H / 2], [0, 0, 1]], dtype=torch.float32
    )

    # Project points to camera coordinates.
    points = pp.point2pixel(
        keypoints.unsqueeze(0).expand(n_frames, -1, -1),
        camera_matrix.unsqueeze(0).expand(n_frames, 3, 3),
        extrinsics=camera_to_object,
    )

    return points
