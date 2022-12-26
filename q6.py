import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cpu")
epochs = 50
batch_size = 6

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
train_x, train_y = train_data['train_data'].astype(np.float32), train_data['train_labels'].astype(np.int32)
valid_x, valid_y = valid_data['valid_data'].astype(np.float32), valid_data['valid_labels'].astype(np.int32)


# Ref: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(train_x.shape[1], 64),
            nn.Sigmoid(),
            nn.Linear(64, train_y.shape[1]),
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    model = Net().to(device)
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
