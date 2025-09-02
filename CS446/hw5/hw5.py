import hw5_utils
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn



## Problem Convolutional Neural Networks

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        - A 2D convolutional layer (torch.nn.Conv2d) with 7 output channels, with kernel size 3
        - A 2D maximimum pooling layer (torch.nn.MaxPool2d), with kernel size 2
        - A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, with kernel size 2
        - A fully connected (torch.nn.Linear) layer with 10 output features

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=7, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(7)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(3)
        self.fc = nn.Linear(12, 10)
        # Please ONLY define the sub-modules here
        


    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        xb = xb.unsqueeze(1)
        xb = self.pool(F.relu(self.bn1(self.conv1(xb))))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = xb.view(xb.size(0), -1)
        xb = self.fc(xb)
        return xb

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    
    net.eval()
    with torch.no_grad():
        train_losses.append(hw5_utils.epoch_loss(net, loss_func, train_dl))
        test_losses.append(hw5_utils.epoch_loss(net, loss_func, test_dl))

    for epoch in range(n_epochs):
        net.train()
        for xb, yb in train_dl:
            hw5_utils.train_batch(net, loss_func, xb, yb, optimizer)

        net.eval()
        with torch.no_grad():
            train_losses.append(hw5_utils.epoch_loss(net, loss_func, train_dl))
            test_losses.append(hw5_utils.epoch_loss(net, loss_func, test_dl))
            
    return train_losses, test_losses





## Problem ResNet

class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return self.relu(x + out)


## Problem Convolutional Neural Networks

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = Block(num_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.block(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x