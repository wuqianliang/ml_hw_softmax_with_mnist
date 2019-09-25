# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Linear embedding of input
class Net_L(nn.Module):
    def __init__(self):
        super(Net_L, self).__init__()
        self.l1 = nn.Linear(784, 10)

    def forward(self, x):
        # flattern (b, 1, 28, 28) --> (n, 784)
        x = x.view(-1, 784)
        return F.log_softmax(self.l1(x), dim=1)

# None linear version embedding of input
class Net_NL(nn.Module):
    def __init__(self):
        super(Net_NL, self).__init__()
        self.l1 = nn.Linear(784, 392)
        self.l2 = nn.Linear(392, 10)

    def forward(self, x):
        # flattern (b, 1, 28, 28) --> (n, 784)
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        # log_softmax = log(softmax(x))
        return F.log_softmax(self.l2(x), dim=1)
    
#model = Net_L().cuda()    
model = Net_NL().cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        # loss
        loss = F.nll_loss(output, target)
        loss.backward()
        # update
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            
            output = model(data)
            # print(output[0][0],target)
            # sum up batch loss
            test_loss += F.nll_loss(output, target).item()
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

for epoch in range(1,5):
    train(epoch)
    test()
