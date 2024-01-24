import torch
import pypose as pp
import tyro
from typing import Dict
from dataclasses import dataclass
from pathlib import Path
import json

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
class FixedLagSmootherConfig:
    horizon: int = 25
    n_keypoints: int = 8
    detector_model_path: str = (
        "outputs/models/fbz72ad3.pth"  # TODO: refactor this to be a config file.
    )

    # Prior config.
    init_pose_mean: pp.SE3 = pp.identity_SE3()
    init_vel_mean: pp.se3 = pp.identity_se3()
    init_pose_covariance = torch.eye(6)
    init_vel_covariance = torch.eye(6)

    # Noise statistics.
    Q_pose = 0.01 * torch.eye(6)
    Q_vel = torch.eye(6)
    R_keypoints = 5 * torch.eye(2)

    # Camera config.
    W: int = 256
    H: int = 256
    camera_intrinsics: torch.Tensor = torch.tensor(
        [[1.9 * 256, 0, 0.5 * 256], [0, 1.9 * 256, 0.5 * 256], [0, 0, 1]]
    )

    object_mass: float = 1.0
    object_inertia: torch.Tensor = torch.eye(3)
    dt: float = 0.1

    # Keypoint config.
    keypoints: torch.Tensor = torch.tensor(UNIT_CUBE_KEYPOINTS).float()


def smoother_config_from_dataset(dataset_path: str, **kwargs):
    # Setup cube mean pose 6 units away from the camera.
    pose_mean = pp.SE3(torch.tensor([0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 1.0]))

    # Setup camera parameters.
    with open(Path(dataset_path) / "metadata.json") as f:
        metadata = json.load(f)

    intrinsics_kubric = metadata["camera"]["K"]
    intrinsics_kubric = torch.tensor(intrinsics_kubric).float()
    camera_intrinsics = (
        torch.diag(torch.tensor([256.0, 256.0, 1.0])) @ intrinsics_kubric
    )

    object_dict = [dd for dd in metadata["instances"] if dd["asset_id"] == "mjc"][0]
    object_scale = object_dict["abs_scale"]
    keypoints = torch.tensor(UNIT_CUBE_KEYPOINTS).float() * object_scale

    return FixedLagSmootherConfig(
        init_pose_mean=pose_mean,
        camera_intrinsics=camera_intrinsics,
        W=256,
        H=256,
        keypoints=keypoints,
        **kwargs
    )


class FixedLagSmoother(torch.nn.Module):
    """
    Implements a fixed-lag smoother for object pose estimation using Theseus.
    """

    def __init__(self, cfg: FixedLagSmootherConfig = FixedLagSmootherConfig()):
        super().__init__()

        self.cfg = cfg

        # Initialize the prior mean/cov.
        self.register_buffer("prior_pose_mean", cfg.init_pose_mean)
        self.register_buffer("prior_pose_covariance", cfg.init_pose_covariance)
        self.register_buffer("prior_vel_mean", cfg.init_vel_mean)
        self.register_buffer("prior_vel_covariance", cfg.init_vel_covariance)

        # Create variables for pose and velocity.
        self.poses = pp.Parameter(
            cfg.init_pose_mean.Log().unsqueeze(0).expand(cfg.horizon + 1, -1).clone()
        )
        self.velocities = pp.Parameter(
            cfg.init_vel_mean.unsqueeze(0).expand(cfg.horizon + 1, -1).clone()
        )

        # Store and register keypoints and camera intrinsics.
        self.register_buffer("camera_intrinsics", cfg.camera_intrinsics)
        self.register_buffer("keypoints", cfg.keypoints)

        self.dt = cfg.dt

    def forward(self, pixel_coordinates: torch.Tensor):
        """
        Returns the residuals used to compute the fixed-lag smoother.
        """

        # Compute prior residuals.
        pose_prior_residual = (self.poses[0].Exp() @ self.prior_pose_mean.Inv()).Log()
        vel_prior_residual = self.velocities[0] - self.prior_vel_mean

        # Compute dynamics residuals.
        pred_pose = self.poses.Exp() @ pp.se3(self.dt * self.velocities).Exp()
        pose_dynamics_residual = (pred_pose[1:] @ pred_pose[:-1].Inv()).Log()

        # Use zero acceleration model.
        velocity_dynamics_residual = self.velocities[1:] - self.velocities[:-1]

        # Compute measurement residuals.
        keypoints_camera_frame = (
            self.poses[:-1].unsqueeze(1).Exp().Act(self.keypoints.unsqueeze(0))
        )

        measurement_residual = pp.reprojerr(
            keypoints_camera_frame,
            pixel_coordinates,
            self.camera_intrinsics,
        )

        return (
            pose_prior_residual,
            vel_prior_residual,
            pose_dynamics_residual,
            velocity_dynamics_residual,
            measurement_residual,
        )
