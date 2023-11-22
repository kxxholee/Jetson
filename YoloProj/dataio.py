import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from videoio import * # => resize_img resize_video 는 Tensor를 반환함
import yaml

def get_labels() -> list[tuple[str, float, float, float, float]]:
    with open('train_settings.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    labels = dataset_config['names']
    return labels

