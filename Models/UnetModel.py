import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
import torch.optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import Models.Unet.unet as unet
from torch.optim.lr_scheduler import StepLR
'''
This is the main script for the U-net. U-net architecture is located in
another location. In this script, the U-net model can be loaded and trained.
The model can then be used to make predictions and plots.
'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestDataset(torch.utils.data.Dataset):
    '''
    Test dataseet class
    '''

    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.float32))
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class TrainDataset(torch.utils.data.Dataset):
    '''
    Train dataset class
    '''

    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.float32))
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        num = np.random.randint(0, 5)
        if num == 0:
            x = x.flip(dims=(1,))
            y = y.flip(dims=(0,))
        elif num == 1:
            x = x.flip(dims=(2,))
            y = y.flip(dims=(1,))
        elif num == 2:
            x = x.flip(dims=(1, 2))
            y = y.flip(dims=(0, 1))
        num = np.random.randint(0, 2)
        if num == 0:
            x = x.transpose(1, 2)
            y = y.transpose(0, 1)
        return x, y

    def __len__(self):
        return len(self.data)


class Model():
    '''
    Model class.
    Initializes U-net.
    Has fit, plot, save, evaluate, predict functions.
    '''

    def __init__(self):
        self.model = unet.UNet(n_channels=2).to(device)
        self.plotsteps = list()
        self.trainloss = list()
        self.testloss = list()
        self.LR = 0

    def fit(self, train, test, trainlabels, testlabels, LR=0.0001, decay=0, batch_size=8, epochstart=0, epochs=50, cropsize=25, momentum=0, dampening=0, stepsize=40):
        self.LR = LR
        criterion = nn.L1Loss()
        optimizer = optim.SGD(self.model.parameters(), lr=LR, weight_decay=decay, momentum=momentum, dampening=dampening)
        scheduler = StepLR(optimizer, step_size=stepsize, gamma=0.1)
        trainset = TrainDataset(train, trainlabels)
        train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                )
        steps = epochstart
        for epoch in range(epochstart, epochs):
            print('Epoch: ', epoch, ', LR: ', scheduler.get_lr())
            running_loss = 0
            batches = 0
            for inputs, labels in train_loader:
                batches = batches+1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = self.model(inputs)
                output = output.squeeze()
                _, dimx, dimy = output.shape
                loss = criterion(output[:, cropsize:dimx-cropsize, cropsize:dimy-cropsize], labels[:, cropsize:dimx-cropsize, cropsize:dimy-cropsize])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            print('Loss :{:.4f} Epoch[{}/{}]'.format(running_loss/batches, epoch, epochs))
            if epoch % 1 == 0:
                train_loss = self.evaluate(train, trainlabels, cropsize)
                test_loss = self.evaluate(test, testlabels, cropsize)
                self.trainloss.append(train_loss)
                self.testloss.append(test_loss)
                self.plotsteps.append(steps)
                steps += 1
                print('Test Loss :{:.4f} Epoch[{}/{}]'.format(test_loss, epoch, epochs))

    def evaluate(self, test, testlabels, cropsize):
        criterion = nn.L1Loss()
        testset = TestDataset(test, testlabels)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False
        )
        self.model.eval()
        running_loss = 0
        batches = 0
        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                output = self.model(inputs)
                output = output.squeeze(dim=0)
                _, dimx, dimy = output.shape
                loss = criterion(output[:, cropsize:dimx-cropsize, cropsize:dimy-cropsize], labels[:, cropsize:dimx-cropsize, cropsize:dimy-cropsize])
                running_loss += loss.item()
                batches += 1
        self.model.train()
        return running_loss/batches

    def predict(self, patches):
        self.model.eval()
        with torch.no_grad():
            output = self.model(patches)
        self.model.train()
        return output

    def save(self, filename):
        filename = 'PreTrainedModels/'+filename
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model = unet.UNet(n_channels=2).to(device)
        filename = 'PreTrainedModels/'+filename
        self.model.load_state_dict(torch.load(filename))

    def plot(self, filename):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
        ax[0].plot(self.plotsteps, self.trainloss)
        ax[0].set_title('Train Loss')
        ax[1].plot(self.plotsteps, self.testloss)
        ax[1].set_title('Test Loss')
        fig.savefig('OutputFiles/'+filename+'.png')
        plt.close(fig)
