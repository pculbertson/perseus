import numpy as np
import gtsam
from typing import List, Optional
from functools import partial


class PoseDynamicsFactor(gtsam.CustomFactor):
    def __init__(
        self,
        pose1: int,
        ang_vel1: int,
        vel1: int,
        pose2: int,
        noise_model: gtsam.noiseModel,
        dt: float,
    ):
        super().__init__(
            noise_model,
            [pose1, ang_vel1, vel1, pose2],
            partial(PoseDynamicsFactor.error_func, dt=dt),
        )
        self.dt = dt

    def error_func(
        this: gtsam.CustomFactor,
        v: gtsam.Values,
        H: Optional[List[np.ndarray]] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        if dt is None:
            raise ValueError("dt must be provided to error_func")

        pose1 = v.atPose3(this.keys()[0])
        ang_vel1 = v.atVector(this.keys()[1])
        vel1 = v.atVector(this.keys()[2])
        pose2 = v.atPose3(this.keys()[3])

        if H:
            # Allocate intermediate jacobians.
            perturbation_jac = np.zeros((6, 6), dtype=np.float64, order="F")
            dpred_dx0 = np.zeros((6, 6), order="F")
            dpred_dtwist = np.zeros((6, 6), order="F")
            drel_dpred = np.zeros((6, 6), order="F")
            drel_dpose2 = np.zeros((6, 6), order="F")

            # Compute predicted pose.
            pose_increment = pose1.Expmap(
                np.concatenate([dt * ang_vel1, dt * vel1]), perturbation_jac
            )
            pred_pose = pose1.compose(pose_increment, dpred_dx0, dpred_dtwist)

            # Compute pose error.
            rel_pose = pred_pose.between(pose2, drel_dpred, drel_dpose2)
            error = rel_pose.Logmap(rel_pose)

            # Implement chain rule
            dlog = rel_pose.LogmapDerivative(rel_pose)
            H[0] = dlog @ drel_dpred @ dpred_dx0

            derr_dtwist = dt * dlog @ drel_dpred @ dpred_dtwist @ perturbation_jac
            H[1] = derr_dtwist[:, :3]
            H[2] = derr_dtwist[:, 3:]

            H[3] = dlog @ drel_dpose2
        else:
            # Compute predicted pose.
            pred_pose = pose1.compose(
                pose1.Expmap(np.concatenate([dt * ang_vel1, dt * vel1]))
            )

            # Compute pose error.
            rel_pose = pred_pose.between(pose2)
            error = rel_pose.Logmap(rel_pose)

        return error


class ConstantVelocityFactor(gtsam.CustomFactor):
    def __init__(
        self,
        vel1: int,
        vel2: int,
        noise_model: gtsam.noiseModel,
    ):
        super().__init__(noise_model, [vel1, vel2], ConstantVelocityFactor.error_func)

    def error_func(
        this: gtsam.CustomFactor, v: gtsam.Values, H: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        vel1 = v.atVector(this.keys()[0])
        vel2 = v.atVector(this.keys()[1])

        if H:
            H[0] = -np.eye(3)
            H[1] = np.eye(3)

        error = vel2 - vel1

        return error


class KeypointProjectionFactor(gtsam.CustomFactor):
    def __init__(
        self,
        body_pose: int,
        noise_model: gtsam.noiseModel,
        camera_intrinsics: gtsam.Cal3_S2,
        keypoint_measurement: np.ndarray,
        point_body_frame: np.ndarray,
    ):
        super().__init__(
            noise_model,
            [body_pose],
            partial(
                KeypointProjectionFactor.error_func,
                camera_intrinsics=camera_intrinsics,
                keypoint_measurement=keypoint_measurement,
                point_body_frame=point_body_frame,
            ),
        )

    def error_func(
        this: gtsam.CustomFactor,
        v: gtsam.Values,
        H: Optional[List[np.ndarray]] = None,
        camera_intrinsics: Optional[gtsam.Cal3_S2] = None,
        keypoint_measurement: Optional[np.ndarray] = None,
        point_body_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        body_pose = v.atPose3(this.keys()[0])

        assert camera_intrinsics is not None
        assert point_body_frame is not None
        assert keypoint_measurement is not None

        if H:
            # Allocate intermediate jacobians.
            dpc_dpose = np.zeros((3, 6), order="F")
            dpc_dpoint = np.zeros((3, 3), order="F")
            dproj_dpose = np.zeros((2, 6), order="F")
            dproj_dcal = np.zeros((2, 5), order="F")
            dproj_dpoint = np.zeros((2, 3), order="F")

            point_camera_frame = body_pose.transformFrom(
                point_body_frame, dpc_dpose, dpc_dpoint
            )

            # Create camera and project point down.
            camera = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(), camera_intrinsics)
            pixel = camera.project(
                point_camera_frame, dproj_dpose, dproj_dpoint, dproj_dcal
            )

            # Implement chain rule.
            H[0] = dproj_dpoint @ dpc_dpose
            # H[0] = dpc_dpose

            error = pixel - keypoint_measurement
            # error = point_camera_frame

        else:
            point_camera_frame = body_pose.transformFrom(point_body_frame)
            camera = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(), camera_intrinsics)
            pixel = camera.project(point_camera_frame)
            error = pixel - keypoint_measurement

        return error
