# perseus
A quick-and-dirty vision-based object tracker for in-hand manipulation.

## Installation
1. Clone the repository
2. Install CUDA 11.8 from the conda channel:
```conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit```
3. Install the package:
```pip install -e .[dev]```
4. Pull+install the kubric submodule:
```
git submodule update --init --recursive
cd kubric
pip install -e .
```
5. Install the ZED SDK from the official website: https://www.stereolabs.com/developers/release/

## Usage
Everything in this repo uses tyro. All instructions here assume you're in the repo root.

To train:
```
# default training values
python perseus/detector/train.py

# to see the help message:
python perseus/detector/train.py -h
```
