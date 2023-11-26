
# implement a simple fully connected network with 2 hidden layers
# and 1 output layer

import torch
import torch.nn as nn
import numpy as np
class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], num_classes: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_d = len(hidden_sizes)

        layers = [] 

        layers.append(nn.ReLU())

        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            # batch normalization
            # dropout
            layers.append(nn.Dropout(0.3))
            layers.append(nn.ReLU())

        # define the output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(self.net(x))

def test():
    # create an instance of the network
    model = SimpleNet(input_dim=28*28, num_classes=10)
    # create a random input tensor
    x = torch.randn(64, 1, 28, 28)
    # get the output of the network
    y = model.forward(x)
    # print the output shape
    print(y.shape)

if __name__ == "__main__":
    test()



