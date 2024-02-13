import numpy as np
import math
import gtsam
from perseus.smoother.factors import PoseDynamicsFactor, ConstantVelocityFactor
import pypose as pp
import torch
from gtsam.symbol_shorthand import X, V, W
from functools import partial

# Generate some random problem data.
np.random.seed(0)
torch.manual_seed(0)

# Create random poses + velocities.
initial_estimate = gtsam.Values()
rand_pose1 = gtsam.Pose3().expmap(np.random.randn(6))
rand_pose2 = gtsam.Pose3().expmap(np.random.randn(6))
rand_vel1 = np.random.randn(3)
rand_ang_vel1 = np.random.randn(3)
rand_vel2 = np.random.randn(3)
rand_ang_vel2 = np.random.randn(3)

# Store these in the init. data.
initial_estimate.insert(X(0), rand_pose1)
initial_estimate.insert(V(0), rand_vel1)
initial_estimate.insert(W(0), rand_ang_vel1)
initial_estimate.insert(X(1), rand_pose2)
initial_estimate.insert(V(1), rand_vel2)
initial_estimate.insert(W(1), rand_ang_vel2)

# Pick a dt.
dt = 1e-1

# TODO(pculbert): refactor tests to match new factor API.

# Create a pose dynamics factor to be tested.
dyn_factor = PoseDynamicsFactor(
    X(0),
    W(0),
    V(0),
    X(1),
    gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])),
    dt,
)

vel_fac = ConstantVelocityFactor(
    V(0),
    V(1),
    gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1, 1e-1, 1e-1])),
)


def flip(x):
    return torch.cat([x[3:], x[:3]])


def pypose_error(x0, w0, v0, x1, dx0, dw0, dv0, dx1, vel_frame="world"):
    """Create PyTorch version of the error func for testing."""
    x0_perturbed = x0 @ pp.se3(flip(dx0)).Exp()
    v0_perturbed = v0 + dv0
    w0_perturbed = w0 + dw0
    x1_perturbed = x1 @ pp.se3(flip(dx1)).Exp()

    if vel_frame == "world":
        v0_perturbed = x0_perturbed.rotation().Inv() @ v0_perturbed

    pred_pose = (
        x0_perturbed @ pp.se3(dt * torch.cat([v0_perturbed, w0_perturbed])).Exp()
    )
    rel_pose = pred_pose.Inv() @ x1_perturbed
    return flip(rel_pose.Log())


def torch_constant_vel_error(v1, v2):
    return v2 - v1


# Create pytorch data.
X0 = initial_estimate.atPose3(X(0))
X1 = initial_estimate.atPose3(X(1))
W0 = initial_estimate.atVector(W(0))
V0 = initial_estimate.atVector(V(0))
W1 = initial_estimate.atVector(W(1))
V1 = initial_estimate.atVector(V(1))


x0 = pp.mat2SE3(X0.matrix().astype(np.float64))
x1 = pp.mat2SE3(X1.matrix().astype(np.float64))
w0 = torch.tensor(W0.astype(np.float64))
v0 = torch.tensor(V0.astype(np.float64))
w1 = torch.tensor(W1.astype(np.float64))
v1 = torch.tensor(V1.astype(np.float64))

# Stand up dummy perturbation variables so we can autodiff easily.
dx0 = torch.zeros(6, dtype=torch.float64)
dw0 = torch.zeros(3, dtype=torch.float64)
dv0 = torch.zeros(3, dtype=torch.float64)
dx1 = torch.zeros(6, dtype=torch.float64)


def test_dynamics_outputs_world_frame():
    """Checks that the error returned by the error_func is correct."""
    gtsam_error = PoseDynamicsFactor.error_func(
        dyn_factor, initial_estimate, dt=dt, vel_frame="world"
    )
    torch_error = pypose_error(x0, w0, v0, x1, dx0, dw0, dv0, dx1, vel_frame="world")
    assert np.allclose(gtsam_error, torch_error.numpy(), atol=1e-6)


def test_dynamics_outputs_with_jacobians_world_frame():
    jac_list = [
        np.zeros((6, 6), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 6), order="F"),
    ]

    gtsam_error = PoseDynamicsFactor.error_func(
        dyn_factor, initial_estimate, jac_list, dt=dt, vel_frame="world"
    )
    torch_error = pypose_error(x0, w0, v0, x1, dx0, dw0, dv0, dx1, vel_frame="world")
    assert np.allclose(gtsam_error, torch_error.numpy(), atol=1e-6)


def test_dynamics_jacobians_world_frame():
    jac_list = [
        np.zeros((6, 6), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 6), order="F"),
    ]

    # Run factor forward to store jacobians in-place in jac_list.
    PoseDynamicsFactor.error_func(
        dyn_factor, initial_estimate, jac_list, dt=dt, vel_frame="world"
    )

    error_world = partial(pypose_error, vel_frame="world")

    # Compute jacobians using autodiff.
    pypose_jacs = pp.func.jacrev(error_world, argnums=(4, 5, 6, 7))(
        x0, w0, v0, x1, dx0, dw0, dv0, dx1
    )

    # Check that the jacobians are correct.
    for i in range(4):
        assert np.allclose(jac_list[i], pypose_jacs[i].numpy(), atol=1e-6)


def test_dynamics_outputs_body_frame():
    """Checks that the error returned by the error_func is correct."""
    gtsam_error = PoseDynamicsFactor.error_func(
        dyn_factor, initial_estimate, dt=dt, vel_frame="body"
    )
    torch_error = pypose_error(x0, w0, v0, x1, dx0, dw0, dv0, dx1, vel_frame="body")
    assert np.allclose(gtsam_error, torch_error.numpy(), atol=1e-6)


def test_dynamics_outputs_with_jacobians_body_frame():
    jac_list = [
        np.zeros((6, 6), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 6), order="F"),
    ]

    gtsam_error = PoseDynamicsFactor.error_func(
        dyn_factor, initial_estimate, jac_list, dt=dt, vel_frame="body"
    )
    torch_error = pypose_error(x0, w0, v0, x1, dx0, dw0, dv0, dx1, vel_frame="body")
    assert np.allclose(gtsam_error, torch_error.numpy(), atol=1e-6)


def test_dynamics_jacobians_world_frame():
    jac_list = [
        np.zeros((6, 6), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 3), order="F"),
        np.zeros((6, 6), order="F"),
    ]

    # Run factor forward to store jacobians in-place in jac_list.
    PoseDynamicsFactor.error_func(
        dyn_factor, initial_estimate, jac_list, dt=dt, vel_frame="body"
    )

    error_world = partial(pypose_error, vel_frame="body")

    # Compute jacobians using autodiff.
    pypose_jacs = pp.func.jacrev(error_world, argnums=(4, 5, 6, 7))(
        x0, w0, v0, x1, dx0, dw0, dv0, dx1
    )

    # Check that the jacobians are correct.
    for i in range(4):
        assert np.allclose(jac_list[i], pypose_jacs[i].numpy(), atol=1e-6)


def test_constant_vel_outputs():
    gtsam_error = ConstantVelocityFactor.error_func(vel_fac, initial_estimate)
    torch_error = torch_constant_vel_error(v0, v1)

    assert np.allclose(gtsam_error, torch_error.numpy(), atol=1e-6)


def test_constant_vel_jacobians():
    jac_list = [
        np.zeros((3, 3), order="F"),
        np.zeros((3, 3), order="F"),
    ]

    # Run factor forward to store jacobians in-place in jac_list.
    ConstantVelocityFactor.error_func(vel_fac, initial_estimate, jac_list)

    # Compute jacobians using autodiff.
    torch_jacs = pp.func.jacrev(torch_constant_vel_error, argnums=(0, 1))(v0, v1)

    # Check that the jacobians are correct.
    for i in range(2):
        assert np.allclose(jac_list[i], torch_jacs[i].numpy(), atol=1e-6)
