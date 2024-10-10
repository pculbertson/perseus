# perseus
[NOTE] Oct. 9, 2024. This branch holds the state of the repo before unused information was pruned away for the code release of the DROP paper.

A quick-and-dirty vision-based object tracker for in-hand manipulation.

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
6. For data generation, there are some additional assets that must be downloaded and placed in the `data_generation/assets` folder: GSO, HDRI_haven, and KuBasic. The assets as well as accompanying json files should be downloaded there, which you can do by running the appropriate download scripts in `kubric/kubric/scripts` and sorting through the results. You need to modify the `data_dir` field of these json files to point to the correct path - we recommend using an absolute path here to avoid confusion.
7. Install the ZED SDK from the official website: https://www.stereolabs.com/developers/release/

## Usage
Everything in this repo uses tyro. All instructions here assume you're in the repo root.

To train:
```
# default training values
python perseus/detector/train.py

# to see the help message:
python perseus/detector/train.py -h
```

To generate data, run the following in the `data_generation` directory:
```
# generates 2.5k videos each with 24 frames
python generate_all_videos.py

# generates labels from all the frames
python generate_and_label_keypoints.py --job-dir <generated_data_dir>
```
For example, `generated_data_dir` might look like
```
/path/to/perseus/data/2024-08-19_11-23-17
```
IMPORTANT: before running `generate_and_label_keypoints.py`, create a file named `mjc_keypoints.json` in `<generated_data_dir>` with the following contents:
```
[[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
 [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
```
This will place the keypoints at the corners of the cube.

To merge multiple datasets, edit the paths to the hdf5 files you want merged in the `merge_hdf5.py` script in the `data` subdirectory, and run:
```
python /path/to/perseus/data/merge_hdf5.py
```
