from dataclasses import dataclass
from typing import Tuple

import kornia
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# #################### #
# CUSTOM AUGMENTATIONS #
# #################### #

NUM_RGB_CHANNELS = 3
DEPTH_CHANNEL_INDEX = 3


class DepthBiasAugmentation(nn.Module):
    """Randomly samples a bias for depth images."""

    def __init__(self, dev: float = 0.02, p_bias: bool = 0.5, cube_scale: float = 0.035) -> None:
        """Initialize the augmentation.

        Args:
            dev: The amount of one-sided deviation in the bias to uniformly sample (0.0 +/- dev).
            cube_scale: The scale of the cube in the depth image.
            p_bias: The probability of sampling biases.
        """
        super().__init__()
        self.dev = dev
        self.p_bias = p_bias
        self.cube_scale = cube_scale

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the depth image.

        Args:
            depth_image: The depth image to augment, shape=(..., H, W).

        Returns:
            depth_image: The augmented depth image.
        """
        scaled_depth_image = self.cube_scale * depth_image

        # uniformly sample biases over the range [-dev, dev]
        bias_dev_mask = F.dropout(torch.ones_like(scaled_depth_image), p=self.p_bias, training=self.training)
        bias_dev = self.dev * bias_dev_mask * 2 * (torch.rand_like(scaled_depth_image) - 0.5)

        # adding the bias to the depth image
        new_depth_image = scaled_depth_image + bias_dev
        return new_depth_image / self.cube_scale


class DepthGaussianNoiseAugmentation(nn.Module):
    """Adds Gaussian noise to depth images."""

    def __init__(self, std: float = 0.005, cube_scale: float = 0.035) -> None:
        """Initialize the augmentation.

        Args:
            std: The standard deviation of the Gaussian noise to add to the (scaled) depth image.
            cube_scale: The scale of the cube in the depth image.
        """
        super().__init__()
        self.std = std
        self.cube_scale = cube_scale

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the depth image.

        Args:
            depth_image: The depth image to augment, shape=(..., H, W).

        Returns:
            depth_image: The augmented depth image.
        """
        scaled_depth_image = self.cube_scale * depth_image
        noise = self.std * torch.randn_like(scaled_depth_image)
        return (scaled_depth_image + noise) / self.cube_scale


class DepthPlaneAugmentation(nn.Module):
    """Randomly samples near and far cutoff planes for depth images."""

    def __init__(
        self,
        near: bool = True,
        near_mean: float = 0.1,
        near_dev: float = 0.05,
        p_near: float = 0.5,
        far: bool = True,
        far_mean: float = 0.5,
        far_dev: float = 0.05,
        p_far: float = 0.5,
        cube_scale: float = 0.035,
    ) -> None:
        """Initialize the augmentation.

        Args:
            near: Whether to sample near plane cutoffs.
            near_mean: The mean of the near plane cutoff.
            near_dev: The deviation of the near plane cutoff.
            p_near: The probability of sampling near plane cutoffs.
            far: Whether to sample far plane cutoffs.
            far_mean: The mean of the far plane cutoff.
            far_dev: The deviation of the far plane cutoff.
            p_far: The probability of sampling far plane cutoffs.
            cube_scale: The scale of the cube in the depth image.
        """
        super().__init__()
        self.near = near
        self.near_mean = near_mean
        self.near_dev = near_dev
        self.p_near = p_near

        self.far = far
        self.far_mean = far_mean
        self.far_dev = far_dev
        self.p_far = p_far

        self.cube_scale = cube_scale

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the depth image.

        Args:
            depth_image: The depth image to augment, shape=(..., H, W).

        Returns:
            depth_image: The augmented depth image.
        """
        scaled_depth_image = self.cube_scale * depth_image

        # if near, every pixel in the scaled image that is below the near plane is set to 0.0
        # the near plane is near_mean + a uniformly sampled deviation about 0, where the probability that a deviation is
        # sampled is p_near
        if self.near:
            # uniformly sample deviations over the range [-near_dev, near_dev]
            near_dev_mask = F.dropout(torch.ones_like(scaled_depth_image), p=self.p_near, training=self.training)
            near_dev = self.near_dev * near_dev_mask * 2 * (torch.rand_like(scaled_depth_image) - 0.5)

            # computing near plane and cutting off pixels
            near_plane = self.near_mean + near_dev
            near_mask = scaled_depth_image < near_plane
            scaled_depth_image = torch.where(near_mask, torch.zeros_like(scaled_depth_image), scaled_depth_image)

        # similar logic for the far plane
        if self.far:
            # uniformly sample deviations over the range [-far_dev, far_dev]
            far_dev_mask = F.dropout(torch.ones_like(scaled_depth_image), p=self.p_far, training=self.training)
            far_dev = self.far_dev * far_dev_mask * 2 * (torch.rand_like(scaled_depth_image) - 0.5)

            # computing far plane and cutting off pixels
            far_plane = self.far_mean + far_dev
            far_mask = scaled_depth_image > far_plane
            scaled_depth_image = torch.where(far_mask, torch.zeros_like(scaled_depth_image), scaled_depth_image)

        # unscale the depth image
        new_depth_image = scaled_depth_image / self.cube_scale
        return new_depth_image


# ############################ #
# AUGMENTATION CONFIG + MODULE #
# ############################ #


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for data augmentation."""

    cube_scale: float = 0.035  # the real cube has side lengths 0.035 times smaller than the cube in the datagen

    # #################### #
    # GLOBAL AUGMENTATIONS #
    # #################### #

    # random affine transformation + its parameters
    random_affine: bool = True
    degrees: float = 90
    translate: Tuple[float, float] = (0.1, 0.1)
    scale: Tuple[float, float] = (0.9, 1.5)
    shear: float = 0.1

    # randomly erases a randomly selected rectangle with some probability, replacing values with 0.0. Note that for
    # depth channels, this will be interpreted as a rectangle whose points are "too close" to the camera.
    random_erasing: bool = True

    # ######## #
    # RGB ONLY #
    # ######## #

    planckian_jitter: bool = True

    # color jiggle + its parameters
    color_jiggle: bool = True
    brightness: float = 0.2
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.025

    blur: bool = True

    # ########## #
    # DEPTH ONLY #
    # ########## #

    random_bias: bool = True
    dev_bias: float = 0.02  # the amount of one-sided deviation in the bias to uniformly sample (0.0 +/- 0.05)
    p_bias: float = 0.5  # the probability of sampling biases

    # TODO(ahl): implement some version of the quadratic stereo error model if needed
    # see: https://github.com/stereolabs/zed-sdk/issues/44

    depth_gaussian_noise: bool = True
    std_gaussian_noise: float = 0.005  # the standard deviation of the Gaussian noise to add to the depth image

    random_near_plane: bool = True
    scaled_near_plane_mean: float = 0.1  # after correcting for cube scale, the near plane is at 0.1m
    dev_near_plane: float = 0.05  # the amount of one-sided deviation in the plane to uniformly sample (0.1 +/- 0.05)
    p_near_plane: float = 0.5  # the probability of sampling deviations from the near plane

    random_far_plane: bool = True
    scaled_far_plane_mean: float = 0.5  # after correcting for cube scale, the far plane is at 0.5m
    dev_far_plane: float = 0.05  # the amount of one-sided deviation in the plane to uniformly sample (0.5 +/- 0.05)
    p_far_plane: float = 0.5  # the probability of sampling deviations from the far plane


class KeypointAugmentation(torch.nn.Module):
    """Data augmentation for keypoint detection."""

    def __init__(self, cfg: AugmentationConfig, train: bool = True) -> None:
        """Initialize the augmentation.

        Args:
            cfg: The augmentation configuration.
            train: Whether to apply training or test-time augmentations.
        """
        super().__init__()
        self.cfg = cfg
        self.train = train

        self.global_transforms = []
        self.rgb_transforms = []
        self.depth_transforms = []

        # global augmentations
        if cfg.random_affine:
            self.global_transforms.append(
                kornia.augmentation.RandomAffine(
                    degrees=cfg.degrees,
                    translate=cfg.translate,
                    scale=cfg.scale,
                    shear=cfg.shear,
                )
            )

        if cfg.random_erasing:
            self.global_transforms.append(
                kornia.augmentation.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(2.0, 3.0), same_on_batch=False)
            )
            self.global_transforms.append(
                kornia.augmentation.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.05),
                    ratio=(0.8, 1.2),
                    same_on_batch=False,
                    value=1,
                )
            )

        # RGB augmentations
        if cfg.planckian_jitter:
            self.rgb_transforms.append(kornia.augmentation.RandomPlanckianJitter(mode="blackbody"))

        if cfg.color_jiggle:
            self.rgb_transforms.append(
                kornia.augmentation.ColorJiggle(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                )
            )

        if cfg.blur:
            self.rgb_transforms.append(kornia.augmentation.RandomGaussianBlur((5, 5), (3.0, 8.0), p=0.5))

        # depth augmentations
        if cfg.random_bias:
            self.depth_transforms.append(
                DepthBiasAugmentation(dev=cfg.dev_bias, p_bias=cfg.p_bias, cube_scale=cfg.cube_scale)
            )

        if cfg.depth_gaussian_noise:
            self.depth_transforms.append(DepthGaussianNoiseAugmentation(std=cfg.std_gaussian_noise))

        if cfg.random_near_plane or cfg.random_far_plane:
            self.depth_transforms.append(
                DepthPlaneAugmentation(
                    near=cfg.random_near_plane,
                    near_mean=cfg.scaled_near_plane_mean,
                    near_dev=cfg.dev_near_plane,
                    p_near=cfg.p_near_plane,
                    far=cfg.random_far_plane,
                    far_mean=cfg.scaled_far_plane_mean,
                    far_dev=cfg.dev_far_plane,
                    p_far=cfg.p_far_plane,
                    cube_scale=cfg.cube_scale,
                )
            )

        # creating sequential augmentations
        self.global_transform_op = kornia.augmentation.AugmentationSequential(
            *self.global_transforms, data_keys=["image", "keypoints"]
        )
        self.rgb_transform_op = kornia.augmentation.AugmentationSequential(*self.rgb_transforms, data_keys=["image"])
        self.depth_transform_op = nn.Sequential(*self.depth_transforms)

    def forward(self, images: torch.Tensor, pixel_coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to the images and pixel coordinates.

        We always assume that the first three channels are RGB. If there is a fourth channel, it must be metric depth.
        If there is a fifth channel, it must be a binary segmentation mask for the cube.

        Args:
            images: The images to augment, shape=(..., C, H, W).
            pixel_coordinates: The pixel coordinates to augment.

        Returns:
            images: The augmented images.
            pixel_coordinates: The augmented pixel coordinates.
        """
        B = images.shape[0]

        # apply global transforms (all channels + keypoints)
        coords = pixel_coordinates.reshape(B, -1, 2)
        if self.train:
            if len(self.global_transforms) > 0:
                images, coords = self.global_transform_op(images, coords)

            # apply rgb transforms (only RGB channels)
            if len(self.rgb_transforms) > 0:
                images[..., :NUM_RGB_CHANNELS, :, :] = self.rgb_transform_op(images[..., :NUM_RGB_CHANNELS, :, :])

            # apply depth transforms (only depth channel)
            if len(self.depth_transforms) > 0 and images.shape[-3] > NUM_RGB_CHANNELS:
                images[..., DEPTH_CHANNEL_INDEX, :, :] = self.depth_transform_op(images[..., DEPTH_CHANNEL_INDEX, :, :])

        # always normalize pixel coordinates
        coords = kornia.geometry.conversions.normalize_pixel_coordinates(coords, images.shape[-2], images.shape[-1])
        return images, coords.reshape(B, -1)
