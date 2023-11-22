from ultralytics import YOLO
import torch

train_path = "path"
val_path = "path"

model = YOLO('yolov8m', cfg='DataInfo.yaml')

epochs = 50
batch_size = 16

train_data = model.datasets(train_path)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters())

device = "cuda" if torch.cuda.is_available() else 'cpu'
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        imgs, targets = batch
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)

        loss = model.loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.save('yolov8_trained.pth')
