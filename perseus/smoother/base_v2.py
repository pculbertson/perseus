from dataclasses import dataclass
from functools import partial
from perseus.smoother.utils import (
    KeypointConfig,
    keypoint_measurement,
    zero_acc_euler_dynamics,
    get_keypoint_config_from_dataset,
)
import pypose as pp
import torch
from typing import Callable


class RigidBodyTrajectory(torch.nn.Module):
    def __init__(
        self,
        init_pose: pp.SE3_type,
        init_vel: pp.se3_type,
        dynamics: Callable,
        measurement: Callable,
        horizon: int,
    ):
        super().__init__()
        self.horizon = horizon

        self.dynamics = dynamics
        self.measurement = measurement

        # Create variables for pose and velocity.
        self.poses = pp.Parameter(
            init_pose.unsqueeze(0).expand(self.horizon + 1, -1).clone()
        )

        self.velocities = pp.Parameter(
            init_vel.unsqueeze(0).expand(self.horizon + 1, -1).clone()
        )

    def forward(
        self, pose_mean: pp.SE3_type, vel_mean: pp.se3_type, measurement: torch.Tensor
    ):
        """
        Returns the residuals used to compute the fixed-lag smoother.
        """
        assert len(pose_mean.lshape) == 0
        assert len(vel_mean.lshape) == 0
        assert measurement.shape[0] == self.horizon

        # Return initial pose/velocity for prior residual.
        pose_residual = (self.poses[0] @ pose_mean.Inv()).Log()
        velocity_prior = self.velocities[0] - vel_mean

        # Return dynamics evaluations for dyn. residual.
        pred_pose, pred_vel = self.dynamics(self.poses[:-1], self.velocities[:-1])
        pose_dyn_residual = (pred_pose @ self.poses[1:].Inv()).Log()
        velocity_dyn_residual = pred_vel - self.velocities[1:]

        # Return predicted measurements for measurement residual.
        pred_measurement = self.measurement(self.poses[:-1])
        meas_residual = pred_measurement - measurement

        return (
            pose_residual.tensor(),
            velocity_prior,
            pose_dyn_residual.tensor(),
            velocity_dyn_residual,
            meas_residual,
        )


@dataclass
class SmootherConfig:
    horizon: int = 20

    init_pose_mean: pp.SE3 = pp.SE3(torch.tensor([0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 1.0]))
    init_pose_covariance: torch.Tensor = torch.eye(6)
    init_vel_mean: pp.se3 = pp.identity_se3()
    init_vel_covariance: torch.Tensor = torch.eye(6)

    Q_pose: torch.Tensor = torch.eye(6) * 0.01
    Q_vel: torch.Tensor = 10 * torch.eye(6)

    R: torch.Tensor = 35 * torch.eye(2)


class FixedLagSmoother:
    def __init__(self, cfg: SmootherConfig, dynamics: Callable, measurement: Callable):
        super().__init__()

        self.trajectory = RigidBodyTrajectory(
            cfg.init_pose_mean,
            cfg.init_vel_mean,
            dynamics,
            measurement,
            cfg.horizon,
        )

        self.horizon = cfg.horizon
        self.cfg = KeypointConfig

        # Store prior parameters.
        self.prior_pose_mean = cfg.init_pose_mean.clone()
        self.prior_pose_covariance = cfg.init_pose_covariance.clone()
        self.prior_vel_mean = cfg.init_vel_mean.clone()
        self.prior_vel_covariance = cfg.init_vel_covariance.clone()

        # Store noise covariances.
        self.Q_pose = cfg.Q_pose.clone()
        self.Q_vel = cfg.Q_vel.clone()
        self.R = cfg.R.clone()

        # Stand up Gauss-Newton optimizer.
        self.optimizer = pp.optim.LM(self.trajectory)
        self.weights = (
            torch.linalg.inv(self.prior_pose_covariance),
            torch.linalg.inv(self.prior_vel_covariance),
            torch.linalg.inv(self.Q_pose),
            torch.linalg.inv(self.Q_vel),
            torch.linalg.inv(self.R),
        )

        dummy_measurement = measurement(cfg.init_pose_mean.unsqueeze(0))
        self.measurements = torch.empty(self.horizon, *dummy_measurement.shape[1:])
        self.num_measurements = 0

    def step(self):
        if self.num_measurements < self.horizon:
            # Raise warning -- not enough measurements for meaningful step.
            print("Warning: not enough measurements for meaningful step. Skipping.")
            return

        # Take Gauss-Newton step.
        self.optimizer.step(
            {
                "pose_mean": self.prior_pose_mean,
                "vel_mean": self.prior_vel_mean,
                "measurement": self.measurements,
            },
            weight=self.weights,
        )

    def update(self, new_measurement: torch.Tensor):
        """
        Update the smoother with a new measurement.
        """
        # Shape checks.
        assert new_measurement.shape == self.measurements.shape[1:]

        if self.num_measurements < self.horizon:
            self.measurements[self.num_measurements] = new_measurement

            self.num_measurements += 1

        else:
            # Shift measurements.
            self.measurements[:-1] = self.measurements[1:].clone()
            self.measurements[-1] = new_measurement

            # # Update prior.
            self.prior_pose_mean = self.trajectory.poses[1].clone()
            self.prior_vel_mean = self.trajectory.velocities[1].clone()

            # # Shift trajectory.
            # self.trajectory.poses = pp.Parameter(
            #     torch.cat(
            #         (
            #             self.trajectory.poses[1:].clone(),
            #             self.trajectory.poses[-1:].clone(),
            #         )
            #     )
            # )
            # self.trajectory.velocities = pp.Parameter(
            #     torch.cat(
            #         (
            #             self.trajectory.velocities[1:].clone(),
            #             self.trajectory.velocities[-1:].clone(),
            #         )
            #     )
            # )


if __name__ == "__main__":
    # Tests.

    # Instantiate a rigid body trajectory.
    init_pose = pp.SE3(torch.tensor([0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 1.0]))
    init_vel = pp.se3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    horizon = 20
    dt = 1.0 / 30.0
    keypoint_cfg = get_keypoint_config_from_dataset(
        "data/2024-01-12_17-21-39/ff2187eb-55ac-4b05-bf31-213656fcd17a"
    )

    dynamics = partial(zero_acc_euler_dynamics, dt=dt)
    measurement = partial(keypoint_measurement, cfg=keypoint_cfg)
    traj = RigidBodyTrajectory(init_pose, init_vel, dynamics, measurement, horizon)

    # Test forward pass.
    (
        pose_prior,
        velocity_prior,
        pred_pose,
        pred_vel,
        pred_measurement,
    ) = traj()
