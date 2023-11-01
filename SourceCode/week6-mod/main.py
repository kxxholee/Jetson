import argparse
from functools import partial

from time import time

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.utils.prune as prune

from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import baseDataset, get_dataloader
from network import *

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./MNIST")
parser.add_argument("--model-type", type=str, default="A", choices=["A", "B"])
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"])
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--prune", type=bool, default=False)

def get_optimizer(name:str):
    if name == "SGD":
        return partial(SGD, weight_deacy=0.005, momentum=0.9)
    elif name == "Adam":
        return partial(Adam, weight_decay=0.005)
    else:
        raise NotImplementedError(name)
    
def get_model(type:str):
    assert type in ["A", "B"]
    if type == "A":
        return modelA()
    else:
        return modelB()

if __name__ == "__main__":
    opt = parser.parse_args()

    # device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    print("Using Device : ", device_name)
    print("cuDNN Enabled : ", torch.backends.cudnn.enabled)
    print("Dataset From : ", opt.root)
    print("Model Name : ", opt.model_type)
    print("Using Optimizer : ", opt.optimizer)
    
    # dataset
    train_loader, val_loader = get_dataloader(opt.root, opt.batch_size, opt.num_workers)

    # network
    net = get_model(opt.model_type).to(DEVICE)

    # criterion, optimizer
    criterion = CrossEntropyLoss()
    optimizer_type = get_optimizer(opt.optimizer)
    optimizer = optimizer_type(net.parameters(), lr=opt.lr)

    # do
    for epoch in range(1, opt.epochs+1):

        time_checkpoint = 0.0

        # Train mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        net.train() # set network to training mode
        time_checkpoint = time()
        for inputs, labels in tqdm(train_loader, ncols=80, ascii=" 0123456789#"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss +=  loss.item()

        time_train = time() - time_checkpoint
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # pruning
        if opt.prune and opt.model_type == "A" and isinstance(module, nn.Conv2d):
            for name, module in tqdm(net.named_modules(), ncols=60, ascii=" #"):
                prune.l1_unstructured(module, name="weight", amount=0.2)
        elif opt.prune and opt.model_type == "B" and isinstance(module, DepthwiseConv):
            for name, module in tqdm(net.named_modules(), ncols=60, ascii=" #"):
                prune.ln_structured(module.depthwise, name="weight", amount=0.2, n=2, dim=0)

        # eval mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        net.eval()
        time_checkpoint = time()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, ncols=80, ascii=" 0123456789#"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        time_test = time() - time_checkpoint

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch [{epoch}/{opt.epochs}]")
        print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train time: {time_train}")
        print(f"    Val   Loss: {val_loss  :.4f}, Val   Accuracy: {val_accuracy  :.2f}%, Val   time: {time_test}")
        