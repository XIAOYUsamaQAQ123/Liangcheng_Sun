import torch
import torch.optim as optim
from hw4_utils import contour_torch, XOR_data
from hw4 import XORNet, fit

import matplotlib.pyplot as plt

X, Y = XOR_data()

net = XORNet()
optimizer = optim.Adam(net.parameters(), lr=0.1)

losses = fit(net, optimizer, X, Y, n_epochs=5000)

contour_torch(-2, 2, -2, 2, net)


plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
