import os
import cv2
import numpy as np
import time
import argparse

import torch


def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)


def get_args(tag=None):
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird"""
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=tag,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=84,
        help="The common width and height for all images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="The number of images per batch"
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["sgd", "adam"], default="adam"
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.2)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=1000000)
    parser.add_argument(
        "--replay_memory_size",
        type=int,
        default=5000,
        help="Number of epoches between testing phases",
    )

    parser.add_argument("--saved_path", type=str, default="outputs")
    parser.add_argument("--log_path", type=str, default="outputs")

    args = parser.parse_args()

    output = (
        "outputs/"
        + (args.tag + "/" if args.tag else "")
        + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    )
    os.makedirs(output, exist_ok=True)
    print(f"output: {output}")

    args.saved_path = output
    return args


def get_device():

    # 检查 MPS 是否可用
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available !!!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available !!!")
    else:
        device = torch.device("cpu")
        print("CPU used !!!")

    # device = torch.device("cpu")
    return device


if __name__ == "__main__":
    opt = get_args()
