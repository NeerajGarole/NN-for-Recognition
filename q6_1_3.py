import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


device = torch.device("cpu")
epochs = 20
batch_size = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Flatten())

        self.fc1 = nn.Linear(1600, 48)
        self.s1 = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 16)
        self.s2 = nn.Sigmoid()
        self.fc3 = nn.Linear(48, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.s1(x)
        x = self.fc2(x)
        x = self.s2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = CNN_CIFAR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    acc = []
    losses = []
    for itr in range(epochs):
        acc_train = 0
        total_loss = 0
        count = 0
        for (x_, y_) in train_loader:
            count = count + 1
            op = model(x_.to(device))
            y_ = y_.to(device)
            loss = criterion(op, y_)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = total_loss + loss.item()
            _, pred = torch.max(op, 1)
            acc_train += pred.eq(y_).sum().item()

        acc.append(acc_train * 100.0 / (count * batch_size))
        losses.append(total_loss / count)
        avg_acc = acc_train * 100.0 / (count * batch_size)
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss / count, avg_acc))

    plt.figure()
    plt.plot(range(len(acc)), acc, label="training acc")
    plt.xlabel("epoch")
    plt.ylabel("Training accuracy")
    plt.xlim(0, len(acc) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(range(len(losses)), losses, label="training")
    plt.xlabel("epoch")
    plt.ylabel("Training loss")
    plt.xlim(0, len(losses) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()