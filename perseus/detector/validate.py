import tyro
from dataclasses import dataclass
import torch
from perseus.detector.models import KeypointCNN, KeypointGaussian
from perseus.detector.data import KeypointDataset
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


@dataclass(frozen=True)
class ValConfig:
    model_path: str = "outputs/models/7j7ecpwl.pth"
    dataset_path: str = "data/2023-12-28_20-34-56/mjc_data.hdf5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_train: bool = False
    output_type: str = "regression"
    save_every: int = 5


def validate(cfg: ValConfig):
    # Load model.
    if cfg.output_type == "gaussian":
        model = KeypointGaussian()
    else:
        model = KeypointCNN()
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval()

    # Create dataloader.
    val_dataset = KeypointDataset(cfg.dataset_path, train=cfg.use_train)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
    )

    model.to(cfg.device)

    # For every image in the validation set, plot the image and the predicted
    # keypoint locations.
    for ii, (pixel_coordinates, object_poses, image) in enumerate(val_dataloader):
        if ii % cfg.save_every == 0:
            plt.close("all")
            # Move to device.
            pixel_coordinates = pixel_coordinates.to(cfg.device).reshape(1, -1, 2)
            object_poses = object_poses.to(cfg.device)
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

            fig, ax = plt.subplots()

            # Plot.
            ax.imshow(image[0].permute(1, 2, 0).cpu().numpy().astype("uint8"))

            # Plot ground truth keypoints with jet colormap.
            jet_colors = plt.cm.jet(np.linspace(0, 1, model.n_keypoints))

            for jj in range(model.n_keypoints):
                ax.scatter(
                    (val_dataset.H / 2) * (pixel_coordinates[0, jj, 0].cpu() + 1),
                    (val_dataset.W / 2) * (pixel_coordinates[0, jj, 1].cpu() + 1),
                    c=jet_colors[jj],
                    alpha=0.8,
                    # use asterisk markers
                    marker="*",
                )
            pred_u = (val_dataset.H / 2) * (
                predicted_pixel_coordinates[0, :, 0].cpu() + 1
            ).detach().cpu().numpy()
            pred_v = (val_dataset.W / 2) * (
                predicted_pixel_coordinates[0, :, 1].cpu() + 1
            ).detach().cpu().numpy()

            if cfg.output_type == "gaussian":
                # Compute sizes of ellipses in pixel coordinates.
                sigma = L[0] @ L[0].T
                sigma_pixels = sigma * ((val_dataset.H / 2) ** 2)

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
                    ax.scatter(
                        pred_u[jj],
                        pred_v[jj],
                        c=jet_colors[jj],
                        alpha=0.8,
                    )
            else:
                for jj in range(model.n_keypoints):
                    ax.scatter(
                        pred_u[jj],
                        pred_v[jj],
                        c=jet_colors[jj],
                        alpha=0.8,
                    )

            # Draw lines between  predicted and ground truth keypoints.
            for jj in range(model.n_keypoints):
                if cfg.output_type == "gaussian":
                    ax.plot(
                        [
                            (val_dataset.H / 2)
                            * (pixel_coordinates[0, jj, 0].cpu() + 1),
                            (val_dataset.H / 2)
                            * (predicted_pixel_coordinates[0, jj, 0].cpu() + 1),
                        ],
                        [
                            (val_dataset.W / 2)
                            * (pixel_coordinates[0, jj, 1].cpu() + 1),
                            (val_dataset.W / 2)
                            * (predicted_pixel_coordinates[0, jj, 1].cpu() + 1),
                        ],
                        c="k",
                        alpha=np.exp(
                            -torch.sqrt(
                                L[0, 2 * jj, 2 * jj] ** 2
                                + L[0, 2 * jj + 1, 2 * jj + 1] ** 2
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    )
                else:
                    ax.plot(
                        [
                            (val_dataset.H / 2)
                            * (pixel_coordinates[0, jj, 0].cpu() + 1),
                            (val_dataset.H / 2)
                            * (predicted_pixel_coordinates[0, jj, 0].cpu() + 1),
                        ],
                        [
                            (val_dataset.W / 2)
                            * (pixel_coordinates[0, jj, 1].cpu() + 1),
                            (val_dataset.W / 2)
                            * (predicted_pixel_coordinates[0, jj, 1].cpu() + 1),
                        ],
                        c="k",
                        alpha=0.9,
                    )

            # Save figure as png.
            plt.savefig(
                f"outputs/figures/val_{ii}.png", bbox_inches="tight", pad_inches=0
            )

            breakpoint()
            if cfg.output_type == "gaussian":
                loss = (
                    -torch.distributions.multivariate_normal.MultivariateNormal(
                        mu, scale_tril=L
                    )
                    .log_prob(pixel_coordinates.reshape(1, -1))
                    .mean()
                )
                print(f"flattened err: {mu-pixel_coordinates.reshape(1, -1)}")
                print(f"reshaped err: {mu.reshape(1, -1, 2)-pixel_coordinates}")
            else:
                loss = torch.nn.SmoothL1Loss(beta=0.25)(
                    pixel_coordinates, predicted_pixel_coordinates.reshape(1, -1, 2)
                )

            print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    tyro.cli(validate)
