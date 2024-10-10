# Perseus

This repository holds the source code for the data collection and cube keypoint tracker used in the paper ["DROP: Dexterous Reorientation via Online Planning"](https://arxiv.org/abs/2409.14562).

## Installation
1. Clone the repository
2. Create a conda environment. If doing data generation, you must be on version 3.10, as Kubric relies on some Blender features which depend on the pinned version `bpy==3.6.0`, which requires python 3.10.
```
conda create -n perseus python=3.10
conda activate perseus
```
3. Install CUDA 11.8 from the conda channel:
```conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit```
4. If doing data generation, pull the kubric submodule:
```
git submodule update --init --recursive
```
5. Install the package:
```pip install -e .[dev]```
6. For data generation, there are some additional assets that must be downloaded and placed in the `data_generation/assets` folder: GSO, HDRI_haven, and KuBasic. The assets as well as accompanying json files should be downloaded there, which you can do by running the appropriate download scripts in `kubric/kubric/scripts` and sorting through the results. You need to modify the `data_dir` field of these json files to point to the correct path - we recommend using an absolute path here to avoid confusion. If encountering issues acquiring these assets, please raise an issue.
7. Install the ZED SDK from the official website: https://www.stereolabs.com/developers/release/

## Usage

This repo holds code for a few different things: (1) data generation for training the keypoint predictor, (2) the keypoint predictor model definition + training/eval code, and (3) factors for the GTSAM factor graph used for downstream pose estimation. All below instructions assume that you are in the repo root.

### Data Generation

The data generation process occurs in a few steps that call separate scripts (this is because the generation process is quite lengthy):
1. Batches of raw data are generated using `kubric`. What this means is that `pybullet` simulations are run that drops the cube asset into a scene along with random other objects, and a simulated video of the scene is rendered. 24 frames from each video are then captured and stored as part of that trajectory.
2. The keypoints for each cube along with other metadata are generated for each video.
3. After generating and labeling the desired quantity of data batches, they will be merged into one dataset.
4. Finally, the datasets will be pruned based on the ratio of pixels corresponding to the cube relative to all the pixels in the image. We call this the segmentation ratio. You can choose a lower and upper bound of acceptable segmentation ratios for the pruning process. This is the dataset which will ultimately be used for training and validation.

Step 1: To generate data, run the following in the `data_generation` directory:
```
# generates 2.5k videos each with 24 frames
# you can stop the script early if needed, it will just produce fewer frames
python generate_all_videos.py
```
This step will generate a directory containing all the generated data, which we refer to as `<generated_data_dir>` below. For instance, it may look like
```
/path/to/perseus/data/2024-08-19_11-23-17
```

Step 2: label the data generated from the first step. Repeat steps 1 and 2 as needed until the desired quantity of data has been generated.
```
# generates labels from all the frames
python generate_and_label_keypoints.py --job-dir <generated_data_dir>
```

Step 3: To merge multiple datasets, edit the paths to the hdf5 files you want merged in the `/path/to/perseus/data/merge_hdf5.py` script in the `data` subdirectory (scroll to the very bottom and edit it directly), and run:
```
python /path/to/perseus/data/merge_hdf5.py
```

Step 4: To prune a merged dataset, edit the paths in the `/path/to/perseus/data/prune_dataset.py` file (scroll to the very bottom and edit it directly), and run:
```
python /path/to/perseus/data/prune_dataset.py
```
After doing this, the path to this dataset will be passed into the training or validation config object below. There are assumed default paths, so if you modify any dataset paths or names, you should double check what you pass into the train and/or validation scripts.

### Training and Evaluating the Predictor

To train with defaults:
```
# default training values
python perseus/detector/train.py

# to see the help message:
python perseus/detector/train.py -h
```
The train script will save a model to the path
```
/path/to/perseus/outputs/models/<wandb_id>.pth
```

To validate a trained model:
```
python perseus/detector/validate.py model_path=<path/to/model.pth> dataset_config.dataset_path=<path/to/pruned/dataset.hdf5>
```
The validation script will not only print numerical information to the terminal, it will also save images useful for qualitative analysis to
```
/path/to/perseus/outputs/figures/<ckpt_name>/sim
```
In particular, for an RGBD model, it will save both the RGB and depth images and overlay the predicted keypoint locations on the image along with the true location.

We also have a script that can run a similar validation procedure on real RGB images (we never refactored it to work for RGBD models, so if you would like to use this feature, you will have to modify the script for your own purposes):
```
python perseus/detector/validate_real.py model_path=<path/to/model.pth> dataset_config.dataset_path=<path/to/pruned/dataset.hdf5>
```
The main difference here is that since there are no ground truth keypoints, you can only see how reasonable the predictions are overlayed on real images.

The better way to evaluate the keypoint detector on real-world data is to use the script `/path/to/perseus/scripts/streaming.py`, which will stream images from a real-world ZED camera while overlaying the keypoint predictions onto it in real time. To properly use this, you should edit the `serial_number` argument at the bottom of the script to match the serial number of the camera you would like to use to stream.
