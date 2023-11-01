import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

def get_image(p:str):
    return Image.open(p).convert("RGB")

class baseDataset(Dataset):
    def __init__(self, root:str="./MNIST", istrain:bool=True):
        super().__init__()
        root = os.path.join(root, "train") if istrain else os.path.join(root, "val")

        data_list = []
        for i in range(10):
            class_index = str(i)
            class_dir = os.path.join(root, class_index)
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                data_list.append((i, img_path))

        self.data_list = data_list

        if istrain:
            self.transform = T.Compose([
                T.RandomCrop(28),
                T.RandomRotation(15),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop(28),
                T.ToTensor()
            ])
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx:int):
        number, img_path = self.data_list[idx]
        img_obj = get_image(img_path)
        img_tensor = self.transform(img_obj)

        return img_tensor, number
    
def get_dataloader(root:str, batch_size:int=32, num_workers:int=32):
    train_dataset, val_dataset = baseDataset(root, True), baseDataset(root, False)
    train_loader    = DataLoader(train_dataset  , batch_size, shuffle=True, num_workers=num_workers)
    val_loader      = DataLoader(val_dataset    , batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader

if __name__ == "__main__":
    pass
