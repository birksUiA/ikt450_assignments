import torch.nn as nn
import torch.nn.functional as F
from torchvision import torch
import functools
import operator


class ConvoModel(nn.Module):

    def __init__(self, image_height=255, image_width=255, channels=3):
        super(ConvoModel, self).__init__()
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def convulayer(inchns, outchns, kernsize):
            layer = nn.Conv2d(inchns, outchns, kernsize)
            return [layer, 
                    nn.ReLU()]

        def dense_layer(input_size, output_size):
            layer = nn.Linear(input_size, output_size)
            return [layer,
                    nn.ReLU()]

        self.cnn = []
        self.cnn += convulayer(3, 32, 5)
        self.cnn += [nn.MaxPool2d(2, 2)]
        self.cnn += convulayer(32, 64, 5)
        self.cnn += [nn.MaxPool2d(2, 2)]
        self.cnn += convulayer(64, 128, 5)
        self.cnn += [nn.MaxPool2d(2, 2)]
        self.cnn += convulayer(128, 256, 3)
        self.cnn += [nn.MaxPool2d(2, 2)]

        self.convo = nn.Sequential(* self.cnn)

        self.dnn = [] 
        self.dnn += dense_layer(self._get_dense_layer_first_input(image_height,
                                                                  image_width,
                                                                  channels),
                                1024)
        self.dnn += dense_layer(1024, 11)
        self.dnn += [nn.Softmax(dim=-1)]
        
        self.dense = nn.Sequential( * self.dnn)

    def forward(self, x):

        conv_output  = self.convo.forward(x)

        flat_output  = torch.flatten(conv_output, start_dim=1) if len(conv_output.shape) == 4 else torch.flatten(conv_output)

        dense_output = self.dense.forward(flat_output) 

        return dense_output
    
    def _get_convo_out_dims(self, image_height, image_width, channels):
        rand_image_like_data = torch.rand(channels, image_height, image_width)
        self.convo.eval()
        data = self.convo.forward(rand_image_like_data)
        self.convo.train(True)
        return data.shape
    
    def _get_dense_layer_first_input(self, image_height, image_width, channels):
        return functools.reduce(lambda x, y: x * y,
                                self._get_convo_out_dims(
                                            image_height, 
                                            image_width, 
                                            channels)
                               )

def main():

    cm = ConvoModel()
    cm.eval()
    test = torch.rand(3, 255, 255)
    print(cm)
    data = cm(test)
    cm.train(True)
    print(data.shape)
    print(data)

if __name__ == "__main__":
    main()
    exit()
else:
    import pytest

def test_get_convo_out_dims():
    image_height, image_width, channels = 255, 255, 3
    testnn = ConvoModel()
    data_shape = testnn._get_convo_out_dims(image_height, image_width, channels)
 
    assert data_shape == torch.rand(256, 21, 21).size()

def test_get_dense_layer_first_input():
    image_height, image_width, channels = 255, 255, 3
    testnn = ConvoModel()
    input_size = testnn._get_dense_layer_first_input(image_height, image_width, channels)
 
    assert input_size == 256 * 21 * 21
