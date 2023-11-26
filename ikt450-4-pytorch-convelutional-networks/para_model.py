import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import torch
import functools
import operator


class ParaModel(nn.Module):

    def __init__(self, image_height=255, image_width=255, channels=3):
        super(ParaModel, self).__init__()
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        def convulayer(inchns, outchns, kernsize):
            layer = nn.Conv2d(inchns, outchns, kernsize)
            return [layer, 
                    nn.ReLU()]

        def dense_layer(input_size, output_size):
            layer = nn.Linear(input_size, output_size)
            return [layer,
                   nn.ReLU()]

        self.tower_1_layers = []
        self.tower_1_layers += convulayer(3, 32, (1, 1))
        self.tower_1 = nn.Sequential(* self.tower_1_layers)
        self.tower_1 = nn.Sequential(*self.tower_1_layers)

        self.tower_2_layers = []
        self.tower_2_layers += convulayer(3, 32, (1, 1))
        self.tower_2_layers += convulayer(32, 32, (3, 3))
        self.tower_2 = nn.Sequential(* self.tower_2_layers)

        self.tower_3_layers = []
        self.tower_3_layers += convulayer(3, 32, (1, 1))
        self.tower_3_layers += convulayer(32, 32, (5, 5))
        self.tower_3 = nn.Sequential(* self.tower_3_layers)

        self.tower_4_layers = []
        self.tower_4_layers += convulayer(3, 32, (1, 1))
        self.tower_4_layers += [nn.MaxPool2d((3, 3))]
        self.tower_4 = nn.Sequential(* self.tower_4_layers)


        self.dnn = [] 
        self.dnn += dense_layer(self._get_dense_layer_first_input(image_height,
                                                                  image_width,
                                                                  channels),
                                1024)
        self.dnn += dense_layer(1024, 11)
        self.dnn += [nn.Softmax(dim=-1)]
        
        self.dense = nn.Sequential( * self.dnn)
    def forward_parallel(self, x):
        y_1 = self.tower_1(x)
        y_1_flat = torch.flatten(y_1, start_dim=1) if len(y_1.shape) == 4 else torch.flatten(y_1)
        y_2 = self.tower_2(x)
        y_2_flat = torch.flatten(y_2, start_dim=1) if len(y_2.shape) == 4 else torch.flatten(y_2)
        y_3 = self.tower_3(x)
        y_3_flat = torch.flatten(y_3, start_dim=1) if len(y_3.shape) == 4 else torch.flatten(y_3)
        y_4 = self.tower_4(x)
        y_4_flat = torch.flatten(y_4, start_dim=1) if len(y_4.shape) == 4 else torch.flatten(y_4)
        return torch.cat((y_1_flat, y_2_flat, y_3_flat, y_4_flat), dim=-1)

    def forward(self, x):
        conv_output  = self.forward_parallel(x)
        dense_output = self.dense.forward(conv_output) 

        return dense_output
    
    def _get_convo_out_dims(self, image_height, image_width, channels):
        rand_image_like_data = torch.rand(channels, image_height, image_width)
        self.eval()
        data = self.forward_parallel(rand_image_like_data)
        self.train(True)
        return data.shape
    
    def _get_dense_layer_first_input(self, image_height, image_width, channels):
        return functools.reduce(lambda x, y: x * y,
                                self._get_convo_out_dims(
                                            image_height, 
                                            image_width, 
                                            channels)
                               )


def main():

    m = ParaModel()
    test = torch.rand(10, 3, 255, 255)
    result = m.forward_parallel(test)
    print(result)
    print(result.shape)


if __name__ == "__main__":
    main()
    exit()
