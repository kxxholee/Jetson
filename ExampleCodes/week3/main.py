import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

MNIST_ROOT = "./datasets/MNIST"

DEV_NAME = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEV_NAME)

print("< Dataset Structure >")
for phase in os.listdir(MNIST_ROOT):
    print(MNIST_ROOT + "/" + phase)
    for dir in os.listdir(os.path.join(MNIST_ROOT, phase)):
        print(f"---------------------/", end="")
        print(dir, f"number : {len(os.listdir(os.path.join(MNIST_ROOT, phase, dir)))}")

# 이미지를 읽는 함수 정의
def get_image(p: str):
    return Image.open(p).convert("L")

class baseDataset(Dataset):
    def __init__(self, root: str, train: bool):
        super().__init__()

        if train:
            self.root = os.path.join(root, "train")
            self.transform = T.Compose([
                T.ToTensor()
            ])
        else:
            self.root = os.path.join(root, "val")
            self.transform = T.Compose([
                T.ToTensor()
            ])
        
        data_list = []
        for i in range(10):
            dir = os.path.join(self.root, str(i))
            for img in os.listdir(dir):
                img_path = os.path.join(dir, img)
                data_list.append((i, img_path))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        number, img_path = self.data_list[idx]

        img_obj = get_image(img_path)
        img_tensor = self.transform(img_obj)

        return img_tensor, number


test_dataset = baseDataset(MNIST_ROOT, False)
print(f"length = {len(test_dataset)}")

sample_tensor, sample_target = test_dataset[10]
C, H, W = sample_tensor.shape
print(f"Channel = {C}, Height = {H}, Width = {W}")
print(f"sample target = {type(sample_target)}, {sample_target}")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim = 1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.head(x)
        return x

net = MLP()
random_input = torch.randn(16, 1, 28, 28)
random_output = net(random_input)
print(random_output.shape)

BATCH_SIZE = 64
train_dataset = baseDataset(MNIST_ROOT, True)
val_dataset = baseDataset(MNIST_ROOT, False)

train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, True)

for images, labels in train_loader:
    print("Train Batch - Image Shape:", images.shape, "Label Shape:", labels.shape)
    break
    # 이미지와 레이블에 대한 추가적인 작업 수행 가능
    # ...

for images, labels in val_loader:
    print("Validation Batch - Image Shape:", images.shape, "Label Shape:", labels.shape)
    break
    # 이미지와 레이블에 대한 추가적인 작업 수행 가능
    # ...

EPOCHS = 10  # 총 에포크 수
LR = 1e-2  # 학습률

net = MLP().to(DEVICE)  # MLP 모델 초기화 및 디바이스 설정
criterion = CrossEntropyLoss()  # 손실 함수 정의
optimizer = SGD(net.parameters(), lr=LR)  # 옵티마이저 정의

# Training loop (학습 반복문)
for epoch in range(EPOCHS):
    train_loss = 0.0  # 훈련 손실 초기화
    train_correct = 0  # 훈련 예측 정확도 초기화
    train_total = 0  # 훈련 데이터 총 개수 초기화
    
    net.train()  # 모델을 훈련 모드로 설정
    
    # Training phase (훈련 단계)
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)  # 입력 데이터를 디바이스로 이동
        labels = labels.to(DEVICE)  # 레이블을 디바이스로 이동
        
        optimizer.zero_grad()  # 그래디언트 초기화
        
        # Forward pass (순전파)
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 손실 계산
        
        # Backward pass and optimization (역전파 및 최적화)
        loss.backward()  # 역전파 수행
        optimizer.step()  # 가중치 업데이트
        
        # Compute accuracy (정확도 계산)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        train_loss += loss.item()  # 훈련 손실 누적
    
    train_loss /= len(train_loader)  # 훈련 손실 평균 계산
    train_accuracy = 100 * train_correct / train_total  # 훈련 정확도 계산
    
    val_loss = 0.0  # 검증 손실 초기화
    val_correct = 0  # 검증 예측 정확도 초기화
    val_total = 0  # 검증 데이터 총 개수 초기화
    
    net.eval()  # 모델을 평가 모드로 설정
    
    # Validation phase (검증 단계)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)  # 입력 데이터를 디바이스로 이동
            labels = labels.to(DEVICE)  # 레이블을 디바이스로 이동
            
            # Forward pass (순전파)
            outputs = net(inputs)
            loss = criterion(outputs, labels)  # 손실 계산
            
            # Compute accuracy (정확도 계산)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_loss += loss.item()  # 검증 손실 누적
    
    val_loss /= len(val_loader)  # 검증 손실 평균 계산
    val_accuracy = 100 * val_correct / val_total  # 검증 정확도 계산
    
    # Print statistics for the current epoch (현재 에포크의 통계 출력)
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
