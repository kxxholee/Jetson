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

class ImageDataset(Dataset):
    def __init__(self, root:str="PATH", labels:list=get_labels(), resize_size:int=416, istrain:bool=True):
        super().__init__()
        root = os.path.join(root, 'train') if istrain else os.path.join(root, 'val')
        self.root = root
        self.labels = labels
        self.resize_size = resize_size

    def __len__(self):
        return len(self.root)

    def __getitem__(self, idx):
        image_path = self.root[idx]
        label = self.labels[idx]
        image_tensor = resize_image(image_path, self.resize_size)
        return image_tensor, label

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, resize_size, max_frames):
        self.video_paths = video_paths
        self.labels = labels
        self.resize_size = resize_size
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_frames = resize_video(video_path, self.resize_size)
        video_frames = video_frames[:self.max_frames]  # 최대 프레임 수 제한

        # 필요한 경우 패딩 추가
        while len(video_frames) < self.max_frames:
            video_frames.append(torch.zeros_like(video_frames[0]))

        video_tensor = torch.stack(video_frames)
        return video_tensor, label


class baseDataset(Dataset):
    def __init__(self, root:str="DEFAULT", istrain:bool=True):
        super().__init__()
        root = os.path.join(root, 'train') if istrain else os.path.join(root, 'val')

        data_list = []
        for i in range(10):
            class_index = str(i)
            class_dir = os.path.join(root, class_index)
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                data_list.append((i, img_path))

        self.data_list = data_list