import torch
import torch.nn as nn
import torch.nn.functional as functions
import torchvision
import numpy as np
import torch.optim as opt
import time

BATCH_SIZE = 100 # informacyjnie
LEARNING_RATE = 0.015
EPOCHS = 5

class SpecifiedRNN(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(SpecifiedRNN, self).__init__()
        self.rnn_unrolled = nn.RNN(i_size, h_size, 4)
        self.dense = nn.Linear(h_size, o_size)
        self.hidden_size = h_size

    def forward(self, x):
        out, hidden = self.rnn_unrolled.forward(x)
        x = self.dense.forward(out[3])
        return functions.log_softmax(x, dim=0)

def rearrange_known_data(train_loader, test_loader):
    repaired_train_data = np.empty((600,100,4,196), dtype="float32")
    repaired_train_target = np.empty((600,100,10), dtype="float32")
    repaired_test_data = np.empty((10000,4,196), dtype="float32")
    repaired_test_target = np.empty((10000,10), dtype="float32")
    iterator = 0
    for (batch_data, batch_target) in train_loader:   
        buffer_d = np.empty((100,4,196))
        buffer_t = np.empty((100,10))
        it_2 = 0
        for instance_data, instance_target in zip(batch_data, batch_target):
            buffer_d[it_2] = np.reshape(instance_data, (4, 196))
            buffer_t[it_2] = np.float_(functions.one_hot(instance_target, 10))
            it_2+=1
        repaired_train_data[iterator] = buffer_d
        repaired_train_target[iterator] = buffer_t
        iterator+=1
    iterator = 0
    for (batch_data, batch_target) in test_loader:
        it_2 = 0
        for instance_data, instance_target in zip(batch_data, batch_target):
            repaired_test_data[it_2] = np.reshape(instance_data, (4, 196))
            repaired_test_target[it_2] = np.float_(functions.one_hot(instance_target, 10))
            it_2+=1
    repaired_train_data = np.swapaxes(repaired_train_data, 1, 2)
    return torch.from_numpy(repaired_train_data), torch.from_numpy(repaired_train_target),\
           torch.from_numpy(repaired_test_data), torch.from_numpy(repaired_test_target)


if __name__ == "__main__":
    net = SpecifiedRNN(196, 64, 10)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=10000, shuffle=True)

    print("Reshaping data...")
    train_data, train_target, test_data, test_target = rearrange_known_data(
        train_loader, test_loader)
    print("Data reshaped.")

    descender = opt.SGD(net.parameters(), lr=LEARNING_RATE)
    start = time.time()

    net.train()
    for i in range(0, EPOCHS):
        batch_flag = 0
        descender.zero_grad()
        for (train_x, train_y) in zip(train_data, train_target):
            descender.zero_grad()
            res = net.forward(train_x)
            loss = functions.cross_entropy(res, train_y)
            loss.backward()
            descender.step()
        print('{}. epoch finished. Final epoch\'s loss: {}'.format(i+1,loss))

    end = time.time()
    print('Training time: {:.2f}s'.format(end-start), )
    net.eval()
    t_losses = []
    correct = 0
    with torch.no_grad():
        for (test_x, test_y) in zip(test_data, test_target):
                res = net.forward(test_x)
                loss = functions.cross_entropy(res, test_y)
                t_losses.append(loss)
                pred = res.data.max(0, keepdim=True)[1]
                correct += pred.eq(test_y.data.argmax()).sum()
    print('Test accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Mean test loss: {}'.format(sum(t_losses)/len(t_losses)))
