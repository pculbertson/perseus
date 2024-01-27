from dataclasses import dataclass
import json
from pathlib import Path
import pypose as pp
import torch


def zero_acc_euler_dynamics(
    x: pp.SE3_type,
    v: pp.se3_type,
    dt: float,
):
    """
    Compute the dynamics of a rigid body in 3D space.
    """
    pred_pose = x @ pp.se3(dt * v).Exp()
    pred_vel = v

    return pred_pose, pred_vel


UNIT_CUBE_KEYPOINTS = [
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
]


@dataclass
class KeypointConfig:
    W: int = 256
    H: int = 256
    camera_intrinsics: torch.Tensor = torch.tensor(
        [[1.9 * 256, 0, 0.5 * 256], [0, 1.9 * 256, 0.5 * 256], [0, 0, 1]]
    )
    keypoints: torch.Tensor = torch.tensor(UNIT_CUBE_KEYPOINTS).float()


def keypoint_measurement(x: pp.SE3_type, cfg: KeypointConfig):
    """
    Compute the measurement of a rigid body in 3D space.
    """
    # Project keypoints onto image plane.
    keypoints_camera_frame = x.unsqueeze(1).Act(cfg.keypoints.unsqueeze(0))

    return pp.point2pixel(keypoints_camera_frame, cfg.camera_intrinsics)


def get_keypoint_config_from_dataset(dataset_path: str):
    # Setup camera parameters.
    with open(Path(dataset_path) / "metadata.json") as f:
        metadata = json.load(f)

    intrinsics_kubric = metadata["camera"]["K"]
    intrinsics_kubric = torch.tensor(intrinsics_kubric).float()
    camera_intrinsics = (
        torch.diag(torch.tensor([256.0, 256.0, 1.0])) @ intrinsics_kubric
    )

    W = H = metadata["flags"]["resolution"]
    object_dict = [dd for dd in metadata["instances"] if dd["asset_id"] == "mjc"][0]
    object_scale = object_dict["abs_scale"]
    keypoints = torch.tensor(UNIT_CUBE_KEYPOINTS).float() * object_scale

    return KeypointConfig(
        W=W,
        H=H,
        camera_intrinsics=camera_intrinsics,
        keypoints=keypoints,
    )
