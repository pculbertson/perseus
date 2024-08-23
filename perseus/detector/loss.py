import numpy as np
import torch


def _gaussian_chol_loss_fn(pred: torch.Tensor, pixel_coordinates: torch.Tensor, n_keypoints: int) -> torch.Tensor:
    """Loss function for the Gaussian keypoint model with 'chol' option.

    Args:
        pred: The predicted parameters of the Gaussian distribution.
        pixel_coordinates: The pixel coordinates of the keypoints.
        n_keypoints: The number of keypoints.

    Returns:
        The loss.
    """
    mu, L = pred
    return -torch.distributions.multivariate_normal.MultivariateNormal(mu, scale_tril=L).log_prob(
        pixel_coordinates
    ).mean() - (n_keypoints / 2) * torch.log(torch.tensor(2 * np.pi))


def _gaussian_diag_loss_fn(pred: torch.Tensor, pixel_coordinates: torch.Tensor, n_keypoints: int) -> torch.Tensor:
    """Loss function for the Gaussian keypoint model with 'diag' option.

    Args:
        pred: The predicted parameters of the Gaussian distribution.
        pixel_coordinates: The pixel coordinates of the keypoints.
        n_keypoints: The number of keypoints.

    Returns:
        The loss.
    """
    mu, sigma = pred
    return -torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma).log_prob(pixel_coordinates).mean() - (
        n_keypoints / 2
    ) * torch.log(torch.tensor(2 * np.pi))
