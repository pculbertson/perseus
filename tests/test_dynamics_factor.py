import numpy as np
import math
import gtsam
from perseus.smoother.factors import DynamicsFactor
import pypose as pp
import torch
from gtsam.symbol_shorthand import X, V, W

# Generate some random problem data.
np.random.seed(0)
torch.manual_seed(0)

# Create random poses + velocities.
initial_estimate = gtsam.Values()
rand_pose1 = gtsam.Pose3().expmap(np.random.randn(6))
rand_pose2 = gtsam.Pose3().expmap(np.random.randn(6))
rand_vel = np.random.randn(3)
rand_ang_vel = np.random.randn(3)

# Store these in the init. data.
initial_estimate.insert(X(0), rand_pose1)
initial_estimate.insert(V(0), rand_vel)
initial_estimate.insert(W(0), rand_ang_vel)
initial_estimate.insert(X(1), rand_pose2)

# Pick a dt.
dt = 1e-1

# Create a factor to be tested.
dyn_factor = DynamicsFactor(
    gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])),
    X(0),
    W(0),
    V(0),
    X(1),
    dt,
)


def flip(x):
    return torch.cat([x[3:], x[:3]])


def pypose_error(x0, w0, v0, x1, dx0, dw0, dv0, dx1):
    """Create PyTorch version of the error func for testing."""
    x0_perturbed = x0 @ pp.se3(flip(dx0)).Exp()
    v0_perturbed = v0 + dv0
    w0_perturbed = w0 + dw0
    x1_perturbed = x1 @ pp.se3(flip(dx1)).Exp()

    pred_pose = (
        x0_perturbed @ pp.se3(dt * torch.cat([v0_perturbed, w0_perturbed])).Exp()
    )
    rel_pose = pred_pose.Inv() @ x1_perturbed
    return flip(rel_pose.Log())


# Create pytorch data.
X0 = initial_estimate.atPose3(X(0))
X1 = initial_estimate.atPose3(X(1))
W0 = initial_estimate.atVector(W(0))
V0 = initial_estimate.atVector(V(0))

x0 = pp.mat2SE3(X0.matrix().astype(np.float64))
x1 = pp.mat2SE3(X1.matrix().astype(np.float64))
w0 = torch.tensor(W0.astype(np.float64))
v0 = torch.tensor(V0.astype(np.float64))

# Stand up dummy perturbation variables so we can autodiff easily.
dx0 = torch.zeros(6, dtype=torch.float64)
dw0 = torch.zeros(3, dtype=torch.float64)
dv0 = torch.zeros(3, dtype=torch.float64)
dx1 = torch.zeros(6, dtype=torch.float64)


def test_dynamics_outputs():
    """Checks that the error returned by the error_func is correct."""
    gtsam_error = dyn_factor.error_func(initial_estimate)
    torch_error = pypose_error(x0, w0, v0, x1, dx0, dw0, dv0, dx1)
    assert np.allclose(gtsam_error, torch_error.numpy(), atol=1e-6)


def test_dynamics_jacobians():
    jac_list = [
        np.zeros((6, 6), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 6), order="F"),
    ]

    # Run factor forward to store jacobians in-place in jac_list.
    dyn_factor.error_func(initial_estimate, jac_list)

    # Compute jacobians using autodiff.
    pypose_jacs = pp.func.jacrev(pypose_error, argnums=(4, 5, 6, 7))(
        x0, w0, v0, x1, dx0, dw0, dv0, dx1
    )

    # Check that the jacobians are correct.
    for i in range(4):
        assert np.allclose(jac_list[i], pypose_jacs[i].numpy(), atol=1e-6)
