"""Modified validation script that uses real data (without GT pose labels)."""

from dataclasses import dataclass

import imageio.v3 as imageio
import matplotlib
import torch
import tyro

from perseus.detector.data import KeypointDatasetConfig
from perseus.detector.models import KeypointCNN

matplotlib.use("Agg")
from pathlib import Path

import kornia
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from perseus import ROOT


@dataclass(frozen=True)
class ValConfig:
    """Validation configuration."""

    model_path: Path = Path(f"{ROOT}/outputs/models/4b8hrqoo.pth")
    in_channels: int = 4  # 3 for RGB, 4 for RGBD
    dataset_cfg: KeypointDatasetConfig = KeypointDatasetConfig(dataset_path=f"{ROOT}/data/real_imgs")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_every: int = 1


def validate(cfg: ValConfig) -> None:  # noqa: PLR0912, PLR0915
    """Validate the model on the real dataset."""
    # create output dirs
    ckpt_name = str(cfg.model_path).split("/")[-1].split(".")[0]
    Path(f"{ROOT}/outputs/figures/{ckpt_name}/real").mkdir(parents=True, exist_ok=True)

    # Load model.
    model = KeypointCNN(num_channels=cfg.in_channels)
    state_dict = torch.load(str(cfg.model_path), weights_only=True)
    for key in list(state_dict.keys()):
        if "module." in key:
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()

    model.to(cfg.device)

    # Load all images, sorted, from data dir.
    image_files = sorted([f for f in Path(cfg.dataset_cfg.dataset_path).glob("*.png")])
    image_files = [f for f in image_files if "segmentation" not in str(f)]

    # For every image in the validation set, plot the image and the predicted keypoint locations.
    for ii, image_file in enumerate(image_files):
        if ii % cfg.save_every == 0:
            plt.close("all")

            # Load image.
            image = Image.open(image_file).convert("RGB")
            image = kornia.utils.image_to_tensor(np.array(image)) / 255.0
            image = image.to(cfg.device).unsqueeze(0)

            print(f"Read image: {image_file}")

            # Center crop to model image size.
            if image.shape[-2:] != (model.H, model.W):
                image = kornia.geometry.transform.resize(image, int(1.8 * model.H))
                image = kornia.geometry.transform.center_crop(image, (model.H, model.W))

            # Forward pass.
            predicted_pixel_coordinates = model(image).reshape(1, -1, 2).detach()

            predicted_pixel_coordinates = kornia.geometry.denormalize_pixel_coordinates(
                predicted_pixel_coordinates, model.H, model.W
            ).cpu()

            print("Forward pass complete.")

            fig, ax = plt.subplots(figsize=(4, 4))

            # Plot (flip image to BGR -> RGB).
            ax.imshow(image[0].permute(1, 2, 0).cpu().numpy())

            # Plot ground truth keypoints with jet colormap.
            jet_colors = plt.cm.jet(np.linspace(0, 1, model.n_keypoints))

            pred_u = predicted_pixel_coordinates[0, :, 0].detach().cpu()
            pred_v = predicted_pixel_coordinates[0, :, 1].detach().cpu()

            for jj in range(model.n_keypoints):
                ax.scatter(
                    pred_u[jj],
                    pred_v[jj],
                    c=jet_colors[jj],
                    alpha=0.8,
                )

            ax.set_title(f"Image {ii} / {len(image_files)}")

            # Save figure as png.
            plt.savefig(f"{ROOT}/outputs/figures/{ckpt_name}/val_{ii}.png")

    # Create gif from images.
    output_path = Path(f"{ROOT}/outputs/figures/{ckpt_name}")

    # Sort image files (without leading zeros).
    image_files = sorted(
        output_path.glob("*.png"),
        key=lambda x: int(x.stem.split("val_")[-1]),
    )

    # images = [imageio.imread(imfile) for imfile in image_files]
    frames = np.stack([imageio.imread(imfile) for imfile in image_files], axis=0)
    imageio.imwrite(f"{ROOT}/outputs/figures/{ckpt_name}/real/val.gif", frames, loop=0, fps=5)


if __name__ == "__main__":
    tyro.cli(validate)
