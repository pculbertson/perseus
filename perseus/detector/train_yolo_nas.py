import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN
from super_gradients.training.metrics import PoseEstimationMetrics
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose import YoloNASPosePostPredictionCallback
from super_gradients.training.transforms.keypoints import (
    KeypointsBrightnessContrast,
    KeypointsHSV,
    KeypointsImageStandardize,
)
from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback, Phase
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.utils.early_stopping import EarlyStop
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from perseus import ROOT
from perseus.detector.data import KeypointDatasetYoloNas


def train(multigpu: bool = True) -> str:
    """Train the YOLO NAS Pose model."""
    if multigpu:
        setup_device(num_gpus=torch.cuda.device_count())

    # transforms
    keypoints_hsv = KeypointsHSV(prob=0.5, hgain=20, sgain=20, vgain=20)
    keypoints_brightness_contrast = KeypointsBrightnessContrast(
        prob=0.5, brightness_range=[0.8, 1.2], contrast_range=[0.8, 1.2]
    )
    # keypoints_mosaic = KeypointsMosaic(prob=0.8)
    keypoints_image_standardize = KeypointsImageStandardize(max_value=255)
    train_transforms = [
        keypoints_hsv,
        keypoints_brightness_contrast,
        # keypoints_mosaic,
        keypoints_image_standardize,
    ]
    val_transforms = [keypoints_image_standardize]

    # datasets
    train_dataset = KeypointDatasetYoloNas(
        data_dir=f"{ROOT}/data/merged/merged.hdf5",
        transforms=train_transforms,
        train=True,
        lazy=True,
        dataset_root=f"{ROOT}/data/merged",
    )
    val_dataset = KeypointDatasetYoloNas(
        data_dir=f"{ROOT}/data/merged/merged.hdf5",
        transforms=val_transforms,
        train=False,
        lazy=True,
        dataset_root=f"{ROOT}/data/merged",
    )
    print("Created datasets!")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=512,
        sampler=DistributedSampler(train_dataset, shuffle=True),
        num_workers=4,
        collate_fn=YoloNASPoseCollateFN(),
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=512,
        sampler=DistributedSampler(val_dataset, shuffle=False),
        num_workers=4,
        collate_fn=YoloNASPoseCollateFN(),
        pin_memory=True,
    )
    print("Created dataloaders!")

    # model
    model = models.get(Models.YOLO_NAS_POSE_N, num_classes=8, pretrained_weights="coco_pose").cuda()
    print("Created model!")

    # training parameters
    post_prediction_callback = YoloNASPosePostPredictionCallback(
        pose_confidence_threshold=0.01,
        nms_iou_threshold=0.7,
        pre_nms_max_predictions=10,
        post_nms_max_predictions=1,
    )
    metrics = PoseEstimationMetrics(
        num_joints=8,
        oks_sigmas=[0.07] * 8,
        max_objects_per_image=1,
        post_prediction_callback=post_prediction_callback,
    )
    visualization_callback = ExtremeBatchPoseEstimationVisualizationCallback(
        keypoint_colors=train_dataset.keypoint_colors,
        edge_colors=train_dataset.edge_colors,
        edge_links=train_dataset.edge_links,
        loss_to_monitor="YoloNASPoseLoss/loss",
        max=True,
        freq=1,
        max_images=16,
        enable_on_train_loader=True,
        enable_on_valid_loader=True,
        post_prediction_callback=post_prediction_callback,
    )
    early_stop = EarlyStop(
        phase=Phase.VALIDATION_EPOCH_END,
        monitor="AP",
        mode="max",
        min_delta=0.0001,
        patience=100,
        verbose=True,
    )
    train_params = {
        "warmup_mode": "LinearBatchLRWarmup",
        "warmup_initial_lr": 1e-8,
        "lr_warmup_epochs": 2,
        "initial_lr": 5e-4,  # TODO(ahl): increase this?
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.05,
        "max_epochs": 100,
        "zero_weight_decay_on_bias_and_bn": True,
        "batch_accumulate": 1,
        "average_best_models": True,
        "save_ckpt_epoch_list": [],
        "loss": "yolo_nas_pose_loss",
        "criterion_params": {
            "oks_sigmas": [0.07] * 8,
            "classification_loss_weight": 0.0,  # we don't care about classification
            "classification_loss_type": "focal",
            "regression_iou_loss_type": "ciou",
            "iou_loss_weight": 2.5,
            "dfl_loss_weight": 0.01,
            "pose_cls_loss_weight": 0.0,  # we don't care about classification
            "pose_reg_loss_weight": 34.0,
            "pose_classification_loss_type": "focal",
            "rescale_pose_loss_with_assigned_score": True,
            "assigner_multiply_by_pose_oks": True,
        },
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.000001},
        "ema": True,
        "ema_params": {"decay": 0.997, "decay_type": "threshold"},
        "mixed_precision": True,
        "sync_bn": False,
        "valid_metrics_list": [metrics],
        "phase_callbacks": [visualization_callback, early_stop],
        "pre_prediction_callback": None,
        "metric_to_watch": "AP",
        "greater_metric_to_watch_is_better": True,
    }

    # train
    CHECKPOINT_DIR = "test_ckpts"
    trainer = Trainer(experiment_name="test", ckpt_root_dir=CHECKPOINT_DIR)
    trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=val_dataloader)

    # getting best model
    best_model_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth")
    print(80 * "=")
    print(f"Best model path: {best_model_path}")
    print(80 * "=")
    return best_model_path


def viz(best_model_path: str, real: bool = True) -> None:
    """Visualize the keypoint predictions of the YOLO NAS Pose model."""
    # read the image and load it as a torch tensor
    best_model = models.get(
        "yolo_nas_pose_n",
        num_classes=8,
        checkpoint_path=best_model_path,
    )

    # loop through every image in {ROOT}/data/sim_imgs
    if real:
        path = f"{ROOT}/data/real_imgs"
    else:
        path = f"{ROOT}/data/sim_imgs"
    imgs = os.listdir(path)
    for img in tqdm(imgs, desc="Visualizing keypoints", total=len(imgs)):
        if not img.endswith(".png"):
            continue
        with open(f"{path}/{img}", "rb") as f:
            image_np = Image.open(f).convert("RGB")
            image = torch.tensor(np.array(image_np)).permute(2, 0, 1)

            # center crop the image to 256x256
            h_crop = 256
            w_crop = 256
            image = image[
                ...,
                image.shape[-2] // 2 - h_crop // 2 : image.shape[-2] // 2 + h_crop // 2,
                image.shape[-1] // 2 - w_crop // 2 : image.shape[-1] // 2 + w_crop // 2,
            ]

            # visualizing keypoint predictions
            res = best_model.predict(image, conf=0.0)
            if len(res.prediction.poses) > 0:
                keypoints = res.prediction.poses[0][..., :2]
                scores = res.prediction.poses[0][..., 2]

                # create a colormap based on the score (from 0 to 1)
                cmap = plt.cm.get_cmap("viridis")
                plt.imshow(image.permute(1, 2, 0))
                plt.scatter(keypoints[:, 0], keypoints[:, 1], c=scores, cmap=cmap)
                cbar = plt.colorbar()
                cbar.set_label("Confidence")
                plt.clim(0, 1)
                plt.title("Keypoint predictions")
                plt.axis("off")
                plt.savefig(f"{path}/val/{img}_keypoints.png")
                plt.close()
            else:
                print("No keypoints detected!")


if __name__ == "__main__":
    best_model_path = train()
    viz(best_model_path)

    # some checkpoints
    # ckpt = "test_ckpts/test/RUN_20240820_161300_337106/ckpt_best.pth"  # first one trained
    # viz(ckpt, real=True)
