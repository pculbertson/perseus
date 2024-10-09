import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import kornia
import matplotlib
import numpy as np
import torch
import tyro
from matplotlib import pyplot as plt
from tqdm import tqdm

from perseus import ROOT
from perseus.detector.augmentations import AugmentationConfig, KeypointAugmentation
from perseus.detector.data import KeypointDatasetConfig, PrunedKeypointDataset
from perseus.detector.models import KeypointCNN

matplotlib.use("Agg")


@dataclass(frozen=True)
class ValConfig:
    """Validation configuration."""

    model_path: str = f"{ROOT}/outputs/models/wzbx1og6.pth"  # RGBD
    batch_size: int = 256 * 8
    dataset_config: KeypointDatasetConfig = KeypointDatasetConfig(
        dataset_path=f"{ROOT}/data/pruned_dataset/pruned.hdf5"
    )
    depth: bool = True
    augmentation_config: AugmentationConfig = AugmentationConfig()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_train: bool = False


def plot_and_save(args: tuple) -> None:
    """Plot and save the image with keypoints and predicted keypoints."""
    image, pixel_coordinate, predicted_pixel_coordinate, batch_index, image_index, output_dir, cfg, n_keypoints = args

    fig, axs = plt.subplots(1, 2 if cfg.depth else 1, figsize=(4, 8))
    axs = [axs] if not isinstance(axs, np.ndarray) else axs

    axs[0].imshow(image[:3, ...].permute(1, 2, 0))
    if cfg.depth:
        axs[1].imshow(image[3, ...], cmap="gray")
    jet_colors = plt.cm.jet(np.linspace(0, 1, n_keypoints))

    for ax in axs:
        # ground truth keypoints
        for k in range(n_keypoints):
            ax.scatter(
                pixel_coordinate[k, 0],
                pixel_coordinate[k, 1],
                color=jet_colors[k],
                alpha=0.8,
                marker="*",
            )

        # predicted keypoints
        for k in range(n_keypoints):
            ax.scatter(
                predicted_pixel_coordinate[k, 0],
                predicted_pixel_coordinate[k, 1],
                color=jet_colors[k],
                alpha=0.8,
            )

        # draw lines between predicted and ground truth keypoints
        for k in range(n_keypoints):
            ax.plot(
                [pixel_coordinate[k, 0], predicted_pixel_coordinate[k, 0]],
                [pixel_coordinate[k, 1], predicted_pixel_coordinate[k, 1]],
                c="k",
                alpha=0.9,
            )

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"val_{batch_index * cfg.batch_size + image_index}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def validate(cfg: ValConfig) -> tuple:  # noqa: PLR0915
    """Validate the model."""
    # Create output directories.
    ckpt_name = str(cfg.model_path).split("/")[-1].split(".")[0]
    output_dir = Path(f"{ROOT}/outputs/figures/{ckpt_name}/sim")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    model = KeypointCNN(num_channels=4 if cfg.depth else 3)
    state_dict = torch.load(str(cfg.model_path), weights_only=True)
    for key in list(state_dict.keys()):
        if "module." in key:
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()

    # Create dataloader.
    val_dataset = PrunedKeypointDataset(cfg.dataset_config, train=cfg.use_train)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
    )
    val_augmentation = KeypointAugmentation(cfg.augmentation_config, train=False)
    model.to(cfg.device)

    # Run validation and collect results.
    plot_args = []
    losses = []
    for i, example in tqdm(
        enumerate(val_dataloader),
        desc="Validation",
        total=len(val_dataloader),
        leave=True,
    ):
        images = example["image"].to(cfg.device)
        pixel_coordinates = example["pixel_coordinates"].to(cfg.device)
        if cfg.depth:
            depth_images = example["depth_image"].to(cfg.device)
            images = torch.cat((images, depth_images[..., None, :, :]), dim=-3)
        images, pixel_coordinates = val_augmentation(images, pixel_coordinates)

        # Forward pass.
        with torch.no_grad():
            predicted_pixel_coordinates = model(images)
            loss = torch.nn.SmoothL1Loss(beta=1.0, reduction="none")(
                pixel_coordinates.reshape(*pixel_coordinates.shape[:-2], -1),
                predicted_pixel_coordinates,
            )
            losses.append(loss)
            loss = loss.mean(dim=-1)

            # reshape
            pixel_coordinates = pixel_coordinates.detach()  # (B, K, 2)
            predicted_pixel_coordinates = predicted_pixel_coordinates.reshape(
                *predicted_pixel_coordinates.shape[:-1], -1, 2
            ).detach()  # (B, K, 2)

            # normalize
            pixel_coordinates = (
                kornia.geometry.denormalize_pixel_coordinates(pixel_coordinates, val_dataset.H, val_dataset.W)
                .cpu()
                .numpy()
            )
            predicted_pixel_coordinates = (
                kornia.geometry.denormalize_pixel_coordinates(predicted_pixel_coordinates, val_dataset.H, val_dataset.W)
                .cpu()
                .numpy()
            )

        # Prepare arguments for plotting (move to CPU here)
        for j, (image, pixel_coordinate, predicted_pixel_coordinate) in enumerate(
            zip(images.cpu(), pixel_coordinates, predicted_pixel_coordinates, strict=False)
        ):
            plot_args.append((image, pixel_coordinate, predicted_pixel_coordinate, i, j))

    # Print loss statistics
    losses = torch.concatenate(losses).reshape(-1)
    print("=" * 80)
    print("Validation Loss")
    print(f"Mean +/- Stdev: {losses.mean()} +/- {losses.std()}")
    print(f"Min: {losses.min()}")
    print(f"Max: {losses.max()}")
    print(f"Median: {torch.median(losses)}")
    print("=" * 80)

    # plot and save a loss histogram with semilogy axis
    plt.hist(losses.cpu().numpy(), bins=100)
    plt.yscale("log")
    plt.savefig(output_dir / "loss_histogram.png")

    return plot_args, model.n_keypoints, output_dir


def main() -> None:
    """Main function."""
    # first, do all validation on GPU
    cfg = tyro.cli(ValConfig)
    plot_args, n_keypoints, output_dir = validate(cfg)

    # second, do all plotting on CPU with multiprocessing on the output data (avoids fork issues with CUDA)
    num_processes = min(mp.cpu_count(), len(plot_args))  # Adjust number of processes
    with mp.Pool(processes=num_processes) as pool:
        full_args = [(arg + (output_dir, cfg, n_keypoints)) for arg in plot_args]
        list(tqdm(pool.imap(plot_and_save, full_args), total=len(full_args), desc="Plotting"))


if __name__ == "__main__":
    main()
