"""Modified validation script that uses real data (without GT pose labels)."""

import tyro
from dataclasses import dataclass
import torch
from perseus.detector.models import KeypointCNN, KeypointGaussian
from perseus.detector.data import KeypointDataset
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from pathlib import Path
import cv2
from torchvision.transforms import Resize, CenterCrop


@dataclass(frozen=True)
class ValConfig:
    model_path: Path = Path("outputs/models/7j7ecpwl.pth")
    dataset_path: Path = Path("data/zed1")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_type: str = "regression"
    save_every: int = 1


def validate(cfg: ValConfig):
    # Load model.
    if cfg.output_type == "gaussian":
        model = KeypointGaussian()
    else:
        model = KeypointCNN()
    model.load_state_dict(torch.load(str(cfg.model_path)))
    model.eval()

    model.to(cfg.device)

    # Load all images, sorted, from data dir.
    image_files = sorted([f for f in cfg.dataset_path.glob("*.png")])
    image_files = [f for f in image_files if "segmentation" not in str(f)]

    print(image_files)

    # For every image in the validation set, plot the image and the predicted
    # keypoint locations.
    for ii, image_file in enumerate(image_files):
        if ii % cfg.save_every == 0:
            plt.close("all")

            # Load image.
            image = cv2.imread(str(image_file))
            image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

            # Center crop to model image size.
            if image.shape[-2:] != (model.H, model.W):
                image = Resize(model.H)(image)
                image = CenterCrop((model.H, model.W))(image)

            image = image.to(cfg.device)

            # Forward pass.
            if cfg.output_type == "gaussian":
                mu, L = model(image)
                predicted_pixel_coordinates = mu.reshape(1, -1, 2).detach()
                predicted_pixel_vars = (
                    torch.diagonal(L @ L.transpose(-1, -2), dim1=1, dim2=2)
                    .reshape(1, -1, 2)
                    .detach()
                )
            else:
                predicted_pixel_coordinates = model(image).reshape(1, -1, 2).detach()

            fig, ax = plt.subplots(figsize=(4, 4))

            # Plot (flip image to BGR -> RGB).
            ax.imshow(
                image[0].permute(1, 2, 0).cpu().numpy().astype("uint8")[:, :, ::-1]
            )

            # Plot ground truth keypoints with jet colormap.
            jet_colors = plt.cm.jet(np.linspace(0, 1, model.n_keypoints))

            pred_u = (model.H / 2) * (
                predicted_pixel_coordinates[0, :, 0].cpu() + 1
            ).detach().cpu().numpy()
            pred_v = (model.W / 2) * (
                predicted_pixel_coordinates[0, :, 1].cpu() + 1
            ).detach().cpu().numpy()

            if cfg.output_type == "gaussian":
                # Compute sizes of ellipses in pixel coordinates.
                sigma = L[0] @ L[0].T
                sigma_pixels = sigma * ((model.H / 2) ** 2)

                for jj in range(model.n_keypoints):
                    # Loop through block diagonal of covariance matrix, plotting confidence ellipses.

                    # Compute eigenvalues and eigenvectors.
                    eig_vals, eig_vecs = torch.linalg.eigh(
                        sigma_pixels[2 * jj : 2 * jj + 2, 2 * jj : 2 * jj + 2],
                    )

                    # Compute angle of rotation.
                    theta = (
                        torch.atan2(eig_vecs[1, 0], eig_vecs[0, 0])
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    # Compute width and height of ellipse.
                    width = 2 * torch.sqrt(eig_vals[0]).detach().cpu().numpy()
                    height = 2 * torch.sqrt(eig_vals[1]).detach().cpu().numpy()

                    # Plot ellipse.
                    # ellipse = Ellipse(
                    #     (pred_u[jj], pred_v[jj]),
                    #     width,
                    #     height,
                    #     angle=np.rad2deg(theta),
                    #     color="b",
                    #     alpha=0.1,
                    # )
                    # ax.add_artist(ellipse)

            else:
                for jj in range(model.n_keypoints):
                    ax.scatter(
                        pred_u[jj],
                        pred_v[jj],
                        c=jet_colors[jj],
                        alpha=0.8,
                    )

            # Save figure as png.
            plt.savefig(f"outputs/figures/val_{ii}.png")

    # Create gif from images.
    import imageio.v3 as imageio

    output_path = Path("outputs/figures")

    # Sort image files (without leading zeros).
    image_files = sorted(
        output_path.glob("*.png"),
        key=lambda x: int(x.stem.split("val_")[-1]),
    )

    images = [imageio.imread(imfile) for imfile in image_files]
    frames = np.stack([imageio.imread(imfile) for imfile in image_files], axis=0)
    imageio.imwrite("outputs/figures/val.gif", frames, loop=0)


if __name__ == "__main__":
    tyro.cli(validate)
