import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



device = torch.device("cpu")
epochs = 52
batch_size = 6

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'].astype(np.float32), train_data['train_labels'].astype(np.int32)
valid_x, valid_y = valid_data['valid_data'].astype(np.float32), valid_data['valid_labels'].astype(np.int32)
k = valid_x.shape[0]
j = train_x.shape[0]
train_x = np.reshape(train_x, (j, 1, 32, 32))
valid_x = np.reshape(valid_x, (k, 1, 32, 32))


# Ref: https://www.kaggle.com/code/shorliu/simple-cnn-to-repcnn-with-pytorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten())

        self.fc1 = nn.Linear(2704, 64)
        self.s1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(48, 32)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc1(x)
        x = self.s1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    train_data_tensor = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data_tensor, batch_size=batch_size, shuffle=True, num_workers=1)
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
            _, y_ = torch.max(y_.to(device), 1)
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
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss/count, avg_acc))

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



