import numpy as np
import gtsam
from typing import List, Optional
from functools import partial


class PoseDynamicsFactor(gtsam.CustomFactor):
    """
    A factor implementing rigid body dynamics between two poses, given
    a linear and angular velocity.
    """

    def __init__(
        self,
        pose1: int,
        ang_vel1: int,
        vel1: int,
        pose2: int,
        noise_model: gtsam.noiseModel,
        dt: float,
        vel_frame: str = "world",
    ):
        """
        Creates a new PoseDynamicsFactor instance. GTSAM has a very
        particular interface for `error_func` that leads to some strange choices here.
        See: https://github.com/borglab/gtsam/blob/develop/python/CustomFactors.md
        for more information on this interface.

        In particular, we need to pass an error function to super().__init__ that
        takes (only) a gtsam.CustomFactor, a Values object, and optionally a list of Jacobians. That means we can't take other arguments like `dt` directly.

        To work around this, we create a closure of the parameterized self.error_func that captures `dt` and then pass that closure to the super() constructor. This is a bit of a hack, but it works.

        Args:
            pose1: The key for the first pose.
            ang_vel1: The key for the angular velocity (in the body frame).
            vel1: The key for the linear velocity (in the world frame).
            pose2: The key for the second pose.
            noise_model: The noise model for the factor.
            dt: The time step for the dynamics.
        """
        assert vel_frame in ["world", "body"], "vel_frame must be 'world' or 'body'."

        # Create a closure of the error function that captures `dt`.
        # We pass the error_func as a class method because the GTSAM
        # optimizers will pass the factor as the `this` argument automatically.
        super().__init__(
            noise_model,
            [pose1, ang_vel1, vel1, pose2],
            partial(PoseDynamicsFactor.error_func, dt=dt, vel_frame=vel_frame),
        )
        self.dt = dt
        self.vel_frame = vel_frame

    def error_func(
        this: gtsam.CustomFactor,
        v: gtsam.Values,
        H: Optional[List[np.ndarray]] = None,
        dt: Optional[float] = None,
        vel_frame: Optional[str] = None,
    ) -> np.ndarray:
        """
        Takes a factor instance and set of values, and computes the residual between an Euler approximation of the dynamics and the next pose. Optionally takes a list of Jacobians that it overwrites in place. We use a hack to allow
        a `dt` parameter to be passed to the factor constructor, which is then
        captured in the closure for this function.

        Args:
            this: The factor instance.
            v: The current values.
            H: The (optional) Jacobians to be overwritten in-place, as a list of np arrays.
            dt: The time step for the dynamics (not optional, but it will always get passed via the closure in the constructor).

        Returns:
            The log map of the residual pose between an Euler approximation of the dynamics and the next pose variable.
        """

        # Unpack values.
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

            if vel_frame == "world":
                # First transform the linear velocity into the body frame.
                dvb_dpose = np.zeros((3, 6), order="F")
                dvb_dvel = np.zeros((3, 3), order="F")

                # A dirty hack: create a trivial rotation transform that can be used to rotate the linear velocity into the body frame.
                world_to_body = gtsam.Pose3(pose1.rotation(), np.zeros(3))
                vel1 = world_to_body.transformTo(vel1, dvb_dpose, dvb_dvel)

            # Compute predicted pose via the exponential map of the velocities.
            pose_increment = pose1.Expmap(
                np.concatenate([dt * ang_vel1, dt * vel1]), perturbation_jac
            )
            pred_pose = pose1.compose(pose_increment, dpred_dx0, dpred_dtwist)

            # Compute pose error.
            rel_pose = pred_pose.between(pose2, drel_dpred, drel_dpose2)
            error = rel_pose.Logmap(rel_pose)

            # Implement chain rule
            dlog = rel_pose.LogmapDerivative(rel_pose)

            H[0] = dlog @ drel_dpred @ dpred_dx0  # derr_dpose1

            # Compute Jacobians for twist, and slice into angular and linear components.
            derr_dtwist = dt * dlog @ drel_dpred @ dpred_dtwist @ perturbation_jac
            H[1] = derr_dtwist[:, :3]  # derr_dang_vel

            if vel_frame == "world":
                # Compute Jacobian for linear velocity in the world frame.
                H[0][:, :3] += derr_dtwist[:, 3:] @ dvb_dpose[:, :3]  # derr_dpose1

                # Compute Jacobian for linear velocity in the world frame
                H[2] = derr_dtwist[:, 3:] @ dvb_dvel  # derr_dvel1

            else:
                H[2] = derr_dtwist[:, 3:]  # derr_dvel1

            H[3] = dlog @ drel_dpose2  # derr_dpose2
        else:
            # Perform forward pass without Jacobian computation for speed.
            if vel_frame == "world":
                vel1 = pose1.rotation().unrotate(vel1)

            pred_pose = pose1 * gtsam.Pose3().Expmap(
                np.concatenate([dt * ang_vel1, dt * vel1])
            )

            # Compute pose error.
            rel_pose = pred_pose.between(pose2)
            error = gtsam.Pose3.Logmap(rel_pose)

        return error


class ConstantVelocityFactor(gtsam.CustomFactor):
    """
    Implements a factor that enforces constant velocity between two time steps.

    See the docstring for `PoseDynamicsFactor` for more information on the
    strange interface for `error_func` in GTSAM.
    """

    def __init__(
        self,
        vel1: int,
        vel2: int,
        noise_model: gtsam.noiseModel,
    ):
        """
        Initializes the factor with the given keys and noise model.
        """
        super().__init__(noise_model, [vel1, vel2], ConstantVelocityFactor.error_func)

    def error_func(
        this: gtsam.CustomFactor, v: gtsam.Values, H: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Returns the difference between two velocities, and optionally computes
        the Jacobians.
        """
        vel1 = v.atVector(this.keys()[0])
        vel2 = v.atVector(this.keys()[1])

        if H:
            H[0] = -np.eye(3)
            H[1] = np.eye(3)

        error = vel2 - vel1

        return error


class KeypointProjectionFactor(gtsam.CustomFactor):
    """
    Implements a factor that computes the difference between a keypoint
    measurement and the projection of the corresponding body-frame point into the camera frame.

    We reimplement this in Python because the GTSAM ProjectionFactors
    de facto assume the pose variable is camera-to-body, not body-to-camera.
    """

    def __init__(
        self,
        body_pose: int,
        noise_model: gtsam.noiseModel,
        camera_intrinsics: gtsam.Cal3_S2,
        keypoint_measurement: np.ndarray,
        point_body_frame: np.ndarray,
        camera_pose: Optional[gtsam.Pose3] = gtsam.Pose3(),
    ):
        """
        Initializes the factor with the given keys, noise model, and problem data.
        See the docstring for `PoseDynamicsFactor` for more information on the
        strange interface for `error_func` in GTSAM.

        Args:
            body_pose: The key for the body pose.
            noise_model: The noise model for the factor.
            camera_intrinsics: The camera intrinsics, as a GTSAM Cal3_S2 object.
            keypoint_measurement: The 2D keypoint measurement.
            point_body_frame: The 3D point in the body frame.
        """
        super().__init__(
            noise_model,
            [body_pose],
            partial(
                KeypointProjectionFactor.error_func,
                camera_intrinsics=camera_intrinsics,
                keypoint_measurement=keypoint_measurement,
                point_body_frame=point_body_frame,
                camera_pose=camera_pose,
            ),
        )

    def error_func(
        this: gtsam.CustomFactor,
        v: gtsam.Values,
        H: Optional[List[np.ndarray]] = None,
        camera_intrinsics: Optional[gtsam.Cal3_S2] = None,
        keypoint_measurement: Optional[np.ndarray] = None,
        point_body_frame: Optional[np.ndarray] = None,
        camera_pose: Optional[gtsam.Pose3] = gtsam.Pose3(),
    ) -> np.ndarray:
        """
        Returns the difference between the keypoint measurement and the projection of the corresponding body-frame point into the camera frame, and optionally computes the Jacobians.

        Args:
            this: The factor instance.
            v: The current values.
            H: The (optional) Jacobians to be overwritten in-place,
                as a list of np arrays.
            camera_intrinsics: The camera intrinsics, as a GTSAM Cal3_S2 object.
            keypoint_measurement: The 2D keypoint measurement.
            point_body_frame: The 3D point in the body frame.
        """
        body_pose = v.atPose3(this.keys()[0])

        # If optional Jacobians are passed.
        if H:
            # Allocate intermediate jacobians.
            dpc_dpose = np.zeros((3, 6), order="F")
            dpc_dpoint = np.zeros((3, 3), order="F")
            dproj_dpose = np.zeros((2, 6), order="F")
            dproj_dcal = np.zeros((2, 5), order="F")
            dproj_dpoint = np.zeros((2, 3), order="F")

            # Transform point into camera frame.
            point_camera_frame = body_pose.transformFrom(
                point_body_frame, dpc_dpose, dpc_dpoint
            )

            # Create camera and project point onto image plane.
            camera = gtsam.PinholeCameraCal3_S2(camera_pose, camera_intrinsics)
            pixel = camera.project(
                point_camera_frame, dproj_dpose, dproj_dpoint, dproj_dcal
            )

            # Implement chain rule.
            H[0] = dproj_dpoint @ dpc_dpose
            error = pixel - keypoint_measurement

        # Otherwise just run forward pass.
        else:
            point_camera_frame = body_pose.transformFrom(point_body_frame)
            camera = gtsam.PinholeCameraCal3_S2(camera_pose, camera_intrinsics)
            pixel = camera.project(point_camera_frame)
            error = pixel - keypoint_measurement

        return error
