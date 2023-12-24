"""Script to generate all videos for the dataset."""

import multiprocessing
import subprocess
import uuid
from tqdm import tqdm
import kubric as kb
from datetime import datetime
import tensorflow as tf

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import logging

logging.getLogger().setLevel(logging.ERROR)

# Setup args.
parser = kb.ArgumentParser()
parser.add_argument(
    "--num_videos", type=int, default=2500, help="Number of videos to generate."
)
parser.add_argument(
    "--num_workers", type=int, default=16, help="Number of workers to use."
)
# TODO(pculbert): Expose all video generationn arguments.


def generate(args):
    """Generate a video."""
    # Create random hash for this video.
    args.job_id = str(uuid.uuid4())

    subprocess.run(
        [
            "python",
            "data_generation/generate_one_video.py",
            "--job-dir",
            f"data/{args.run_id}/{args.job_id}",
        ]
    )


if __name__ == "__main__":
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.run_id = run_id

    arg_list = [args for _ in range(args.num_videos)]

    with multiprocessing.Pool(args.num_workers) as p:
        it = tqdm(
            p.imap(generate, arg_list),
            total=len(arg_list),
            desc="generating",
        )
        list(it)
