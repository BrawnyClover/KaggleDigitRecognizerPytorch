import so_compli
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random

raw_data = so_compli.load_train_data()

class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        self.linear1 = nn.Linear(784, 10)
    def forward(self, x):
        return self.linear1(x)


def visualize(line_idx):
    raw_0 = raw_data[line_idx]
    for idx in range(28):
        linestr = ''
        for idx2 in range(28):
            pixelstr = str(raw_0[idx*28+idx2])
            linestr += ' '*(4-len(pixelstr)) + pixelstr
        print(linestr)


def get_image_tensor(idx):
    pixel_data = raw_data[idx][1:]
    label = raw_data[idx][0]

    pixel_data = np.expand_dims(pixel_data, 0)
    torch_pixel = torch.from_numpy(pixel_data)
    torch_float_pixel_data = torch_pixel.type(torch.FloatTensor)
    torch_float_pixel_data = torch_float_pixel_data / 255

    label = np.expand_dims(label, 0)
    torch_label = torch.from_numpy(label)
    torch_float_label_data = torch_label.type(torch.LongTensor)

    return torch_float_pixel_data, torch_float_label_data


def train_mnist():
    net = MNISTNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)

    for idx in range(1000000):
        x, y = get_image_tensor(random.randint(0, 41999))
        prob = net(x)
        loss = criterion(prob, y)
       # if idx % 500 == 0:
        #    print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net


net = train_mnist()
torch.save(net, 'so_compli.pt')

# visualize(2)
# x, y = get_image_tensor(2)
# print(net(x))
# #print(raw_data[2][0])
# v, i = torch.max(net(x), 1)
# print(v, i)
# print(i.data.numpy()[0])