

from __future__ import division

import os
import pathlib
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary

# def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
#     """Creates a DataLoader for training.
#
#     :param img_path: Path to file containing all paths to training images.
#     :type img_path: str
#     :param batch_size: Size of each image batch
#     :type batch_size: int
#     :param img_size: Size of each image dimension for yolo
#     :type img_size: int
#     :param n_cpu: Number of cpu threads to use during batch generation
#     :type n_cpu: int
#     :param multiscale_training: Scale images to different sizes randomly
#     :type multiscale_training: bool
#     :return: Returns DataLoader
#     :rtype: DataLoader
#     """
#     dataset = ListDataset(
#         img_path,
#         img_size=img_size,
#         multiscale=multiscale_training,
#         transform=AUGMENTATION_TRANSFORMS)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=n_cpu,
#         pin_memory=True,
#         collate_fn=dataset.collate_fn,
#         worker_init_fn=worker_seed_set)
#     return dataloader

# parser = argparse.ArgumentParser(description="hhh")
# parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
# a = parser.parse_args()
# print(a)

list = []
for i in (1, 2, 3, 4):
    list.append(i)
print(list)
