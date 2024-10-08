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
        near_value: float = 0.0,
        far: bool = True,
        far_mean: float = 0.5,
        far_dev: float = 0.05,
        p_far: float = 0.5,
        far_value: float = 0.0,
        cube_scale: float = 0.035,
    ) -> None:
        """Initialize the augmentation.

        Args:
            near: Whether to sample near plane cutoffs.
            near_mean: The mean of the near plane cutoff.
            near_dev: The deviation of the near plane cutoff.
            p_near: The probability of sampling near plane cutoffs.
            near_value: The value to set pixels below the near plane to.
            far: Whether to sample far plane cutoffs.
            far_mean: The mean of the far plane cutoff.
            far_dev: The deviation of the far plane cutoff.
            p_far: The probability of sampling far plane cutoffs.
            far_value: The value to set pixels above the far plane to.
            cube_scale: The scale of the cube in the depth image.
        """
        super().__init__()
        self.near = near
        self.near_mean = near_mean
        self.near_dev = near_dev
        self.p_near = p_near
        self.near_value = near_value

        self.far = far
        self.far_mean = far_mean
        self.far_dev = far_dev
        self.p_far = p_far
        self.far_value = far_value

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
            scaled_depth_image = torch.where(
                near_mask, self.near_value * torch.ones_like(scaled_depth_image), scaled_depth_image
            )

        # similar logic for the far plane
        if self.far:
            # uniformly sample deviations over the range [-far_dev, far_dev]
            far_dev_mask = F.dropout(torch.ones_like(scaled_depth_image), p=self.p_far, training=self.training)
            far_dev = self.far_dev * far_dev_mask * 2 * (torch.rand_like(scaled_depth_image) - 0.5)

            # computing far plane and cutting off pixels
            far_plane = self.far_mean + far_dev
            far_mask = scaled_depth_image > far_plane
            scaled_depth_image = torch.where(
                far_mask, self.far_value * torch.ones_like(scaled_depth_image), scaled_depth_image
            )

        # unscale the depth image
        new_depth_image = scaled_depth_image / self.cube_scale
        return new_depth_image


class RandomTransplantationWithDepth(nn.Module):
    """Performs random transplantation of a patch from one image to another, using depth channels to correctly layer."""

    def __init__(self, p: float = 0.5, lb_seg_ratio: float = 0.02, ub_seg_ratio: float = 0.7) -> None:
        """Initialize the augmentation.

        Args:
            p: The probability of applying the augmentation.
            lb_seg_ratio: The lower bound of the segmentation ratio for the new images.
            ub_seg_ratio: The upper bound of the segmentation ratio for the new images.
        """
        super().__init__()
        self.p = p
        self.lb_seg_ratio = lb_seg_ratio
        self.ub_seg_ratio = ub_seg_ratio

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Performs the augmentation with proper layering of depth channels.

        Args:
            images: The images to augment, also the source of donors for transplantation, shape=(B, C, H, W).

        Returns:
            new_images: The transplanted images, shape=(B, C, H, W).
        """
        # only apply the random transplantation if segmentation images are available and the images are batched
        if images.shape[-3] != 5 or len(images.shape) <= 3:  # noqa: PLR2004
            return images

        rgb_images = images[..., :NUM_RGB_CHANNELS, :, :]  # (B, 3, H, W)
        depth_images = images[..., DEPTH_CHANNEL_INDEX, :, :]  # (B, H, W)
        seg_images = images[..., -1, :, :]  # (B, H, W)

        # for each image in the batch, randomly select a donor image that is different from the current image
        batch_size = images.shape[0]
        donor_indices = (torch.arange(batch_size) + torch.randint(1, batch_size, (batch_size,))) % batch_size
        donor_images = images[donor_indices]  # (B, 5, H, W)

        # [OPTION] in the donor images, mask out the cube
        # [NOTE] this leaves a pretty noticeable black blotch in the image, I prefer something more "realistic"
        # ind_donor_cube = (donor_images[..., -1, :, :] == 1.0)[:, None, :, :]  # (B, 1, H, W)
        # donor_images = torch.where(
        #     ind_donor_cube, torch.tensor(0.0, device=donor_images.device), donor_images
        # )  # (B, 5, H, W)

        # start creating the donor masks - wherever the cube is NOT in the acceptor (original), we want to transplant
        ind_acceptor_cube = seg_images == 1.0  # (B, H, W)
        donor_masks = ~ind_acceptor_cube

        # additionally, any pixel of the donor's depth image that is smaller than the acceptor's depth image wherever
        # there is a cube pixel in the acceptor image should be added to the donor mask
        depth_with_cube_acceptor = depth_images * ind_acceptor_cube  # (B, H, W)
        depth_with_cube_donor = donor_images[..., DEPTH_CHANNEL_INDEX, :, :] * ind_acceptor_cube  # (B, H, W)
        inds_donor_cube_pixel_closer = depth_with_cube_donor < depth_with_cube_acceptor  # (B, H, W)
        donor_masks[inds_donor_cube_pixel_closer] = True  # (B, H, W)

        # remove cube pixels from the donor mask
        ind_donor_cube = donor_images[..., -1, :, :] == 1.0  # (B, H, W)
        donor_masks = torch.where(ind_donor_cube, torch.zeros_like(donor_masks, device=donor_masks.device), donor_masks)

        # create the new transplanted batch
        _new_rgb_images = torch.where(
            donor_masks.unsqueeze(1), donor_images[..., :NUM_RGB_CHANNELS, :, :], rgb_images
        )  # (B, 3, H, W)
        _new_depth_images = torch.where(donor_masks, donor_images[..., DEPTH_CHANNEL_INDEX, :, :], depth_images)[
            :, None, ...
        ]  # (B, 1, H, W)
        _new_seg_images = (1.0 - donor_masks.float()).unsqueeze(1)  # (B, 1, H, W)
        _new_seg_images = torch.where(
            ind_donor_cube & ~ind_acceptor_cube,
            torch.zeros_like(_new_seg_images[:, 0, ...]),
            _new_seg_images[:, 0, ...],
        )[:, None, ...]  # (B, 1, H, W), remove the donor cube pix from the new seg img unless part of the acceptor cube
        _new_images = torch.cat([_new_rgb_images, _new_depth_images, _new_seg_images], dim=1)  # (B, 5, H, W)

        # use the new images if the new segmentation ratio is within the bounds
        new_seg_ratios = torch.mean(_new_seg_images, dim=(-2, -1)).squeeze()  # (B,)
        ind_within_bounds = (new_seg_ratios >= self.lb_seg_ratio) & (new_seg_ratios <= self.ub_seg_ratio)
        new_images = torch.where(ind_within_bounds.view(-1, 1, 1, 1), _new_images, images)  # ugly broadcasting
        return new_images  # (B, 5, H, W)


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

    # random transplantation using depth
    random_transplantation_with_depth: bool = True

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

    random_plasma_shadow: bool = True

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
    near_value: float = 0.0  # the value to set pixels below the near plane to

    random_far_plane: bool = True
    scaled_far_plane_mean: float = 0.5  # after correcting for cube scale, the far plane is at 0.5m
    dev_far_plane: float = 0.05  # the amount of one-sided deviation in the plane to uniformly sample (0.5 +/- 0.05)
    p_far_plane: float = 0.5  # the probability of sampling deviations from the far plane
    far_value: float = 0.0  # the value to set pixels above the far plane to


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

        self.global_transforms_no_kornia = []
        self.global_transforms = []
        self.rgb_transforms = []
        self.depth_transforms = []

        # non-kornia global augmentations
        if cfg.random_transplantation_with_depth and train:
            self.global_transforms_no_kornia.append(RandomTransplantationWithDepth())

        # global augmentations
        if cfg.random_affine and train:
            self.global_transforms.append(
                kornia.augmentation.RandomAffine(
                    degrees=cfg.degrees,
                    translate=cfg.translate,
                    scale=cfg.scale,
                    shear=cfg.shear,
                )
            )

        if cfg.random_erasing and train:
            self.global_transforms.append(
                kornia.augmentation.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(2.0, 3.0), same_on_batch=False)
            )
            self.global_transforms.append(
                kornia.augmentation.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.05),
                    ratio=(0.8, 1.2),
                    same_on_batch=False,
                )
            )

        # RGB augmentations
        if cfg.planckian_jitter and train:
            self.rgb_transforms.append(kornia.augmentation.RandomPlanckianJitter(mode="blackbody"))

        if cfg.color_jiggle and train:
            self.rgb_transforms.append(
                kornia.augmentation.ColorJiggle(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                )
            )

        if cfg.blur and train:
            self.rgb_transforms.append(kornia.augmentation.RandomGaussianBlur((5, 5), (3.0, 8.0), p=0.5))

        if cfg.random_plasma_shadow and train:
            self.rgb_transforms.append(kornia.augmentation.RandomPlasmaShadow())

        # depth augmentations
        if cfg.random_bias and train:
            self.depth_transforms.append(
                DepthBiasAugmentation(dev=cfg.dev_bias, p_bias=cfg.p_bias, cube_scale=cfg.cube_scale)
            )

        if cfg.depth_gaussian_noise and train:
            self.depth_transforms.append(DepthGaussianNoiseAugmentation(std=cfg.std_gaussian_noise))

        if cfg.random_near_plane or cfg.random_far_plane:
            if train:
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
            else:
                self.depth_transforms.append(
                    DepthPlaneAugmentation(
                        near=cfg.random_near_plane,
                        near_mean=cfg.scaled_near_plane_mean,
                        near_dev=cfg.dev_near_plane,
                        p_near=0.0,
                        far=cfg.random_far_plane,
                        far_mean=cfg.scaled_far_plane_mean,
                        far_dev=cfg.dev_far_plane,
                        p_far=0.0,
                        cube_scale=cfg.cube_scale,
                    )
                )  # in val mode, still cutoff near/far planes, but without noise

        # creating sequential augmentations
        self.global_transform_op_no_kornia = nn.Sequential(*self.global_transforms_no_kornia)
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
        leading_coords_shape = pixel_coordinates.shape[:-1]

        # apply global transforms (all channels + keypoints)
        if pixel_coordinates.shape[-1] != 2:  # noqa: PLR2004
            coords = pixel_coordinates.reshape(*leading_coords_shape, -1, 2)
        else:
            if len(pixel_coordinates.shape) == 2:  # noqa: PLR2004
                assert len(images.shape) == 3, "If no batch dim, images must be 3D"  # noqa: PLR2004
            coords = pixel_coordinates

        # do random transplantation first thing
        if len(self.global_transforms_no_kornia) > 0:
            images = self.global_transform_op_no_kornia(images)

        if len(self.global_transforms) > 0:
            images, coords = self.global_transform_op(images, coords)

        # apply rgb transforms (only RGB channels)
        if len(self.rgb_transforms) > 0:
            images[..., :NUM_RGB_CHANNELS, :, :] = self.rgb_transform_op(images[..., :NUM_RGB_CHANNELS, :, :])

        # apply depth transforms (only depth channel)
        if len(self.depth_transforms) > 0 and images.shape[-3] > NUM_RGB_CHANNELS:
            images[..., DEPTH_CHANNEL_INDEX, :, :] = self.depth_transform_op(images[..., DEPTH_CHANNEL_INDEX, :, :])

        # normalize pixel coordinates
        coords = kornia.geometry.conversions.normalize_pixel_coordinates(coords, images.shape[-2], images.shape[-1])

        # if no batch dim, unsqueeze
        if len(images.shape) == 3:  # noqa: PLR2004
            images = images.unsqueeze(0)
            coords = coords.unsqueeze(0)

        return images, coords.reshape(*leading_coords_shape, -1)
