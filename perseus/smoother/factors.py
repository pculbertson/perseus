import numpy as np
import gtsam
from typing import List, Optional


class DynamicsFactor(gtsam.CustomFactor):
    def __init__(
        self,
        noise_model: gtsam.noiseModel,
        pose1: int,
        ang_vel1: int,
        vel1: int,
        pose2: int,
        dt: float,
    ):
        super().__init__(noise_model, [pose1, ang_vel1, vel1, pose2], self.error_func)
        self.dt = dt

    def error_func(
        self, v: gtsam.Values, H: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        pose1 = v.atPose3(self.keys()[0])
        ang_vel1 = v.atVector(self.keys()[1])
        vel1 = v.atVector(self.keys()[2])
        pose2 = v.atPose3(self.keys()[3])

        if H:
            # Allocate intermediate jacobians.
            perturbation_jac = np.zeros((6, 6), dtype=np.float64, order="F")
            dpred_dx0 = np.zeros((6, 6), order="F")
            dpred_dtwist = np.zeros((6, 6), order="F")
            drel_dpred = np.zeros((6, 6), order="F")
            drel_dpose2 = np.zeros((6, 6), order="F")

            # Compute predicted pose.
            pose_increment = pose1.Expmap(
                np.concatenate([self.dt * ang_vel1, self.dt * vel1]), perturbation_jac
            )
            pred_pose = pose1.compose(pose_increment, dpred_dx0, dpred_dtwist)

            # Compute pose error.
            rel_pose = pred_pose.between(pose2, drel_dpred, drel_dpose2)
            error = rel_pose.Logmap(rel_pose)

            # Implement chain rule
            dlog = rel_pose.LogmapDerivative(rel_pose)
            H[0] = dlog @ drel_dpred @ dpred_dx0

            derr_dtwist = self.dt * dlog @ drel_dpred @ dpred_dtwist @ perturbation_jac
            H[1] = derr_dtwist[:, :3]
            H[2] = derr_dtwist[:, 3:]

            H[3] = dlog @ drel_dpose2
        else:
            # Compute predicted pose.
            pred_pose = pose1.compose(
                pose1.Expmap(np.concatenate([self.dt * ang_vel1, self.dt * vel1]))
            )

            # Compute pose error.
            rel_pose = pred_pose.between(pose2)
            error = rel_pose.Logmap(rel_pose)

        return error
