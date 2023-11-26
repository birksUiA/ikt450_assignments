import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from convo_model import ConvoModel
from dataparser import get_sub_set, load_images
from datetime import datetime


optimizers = {"SGD": torch.optim.SGD}


class TrainModel:
    def __init__(
        self,
        model=None,
        loss_function=torch.nn.MSELoss(),
        optimizer_name="SGD",
        epochs=2,
        traning_dataset=None,
        validition_dataset=None,
    ):
        # Get device to allow traning on GPU 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Some assignements
        self.loss_fn = loss_function
        self.opt_name = optimizer_name
        self.epochs = epochs

        # get or assign model
        if model == None: 
            self.m = ConvoModel()
        else:
            self.m = model

        # Assign optimiser
        if self.opt_name == "SGD":
            self.opt = optimizers[optimizer_name](params=self.m.parameters(), lr=0.001)
        else: 
            print("invalid optimiser name")
            raise RunTimeError


        # get or set traning data set and create loader
        if traning_dataset == None:
            tra = get_sub_set(load_images(path="./data_true/training/"), 0.2)
            self.train_dataset = tra
        else:
            self.train_dataset = traning_dataset

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset, shuffle=True, batch_size=32
        )

        # get or set validation data set
        if validition_dataset == None:
            val = get_sub_set(load_images(path="./data_true/validation/"), 0.2)
            self.val_dataset = val
        else:
            self.val_dataset = validition_dataset
        # Containers for traning
        self.train_losses = []
        self.val_losses = []
        self.accuarcies = []
        self.y = []
        self.y_hat = []
        # Gennerate and create folder for saved models. 
        self.model_path = self.create_model_save_path()
        if not os.path.exists(os.path.dirname(os.path.abspath(self.model_path))):
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)))

    def trian_one_epock(self, epoch_idx):
        running_loss = 0.0
        self.m.train(True)
        for i, data in enumerate(tqdm(self.train_dataloader)):
            inputs, labels_idx = data[0].to(self.device), data[1].to(self.device)
            labels = self.create_labels_tensors(labels_idx)
            output = self.m(inputs.to(self.device))
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.opt.step()
            running_loss += loss.item()
        return running_loss

    def validate_one_epoch(self):
        running_loss = 0.0
        self.m.eval()
        with torch.no_grad():
            i = 0
            posetives = 0; 
            for i, data in enumerate(tqdm(self.val_dataset)):
                inp, label = data[0].to(self.device), data[1]
                lapel_tensor = self.create_labels_tensors(label)
                output = self.m(inp)
                loss = self.loss_fn(output, lapel_tensor)
                if torch.argmax(output) == label:
                    posetives += 1
                
                running_loss += loss.item()
                self.y.append(label)
                self.y_hat.append(torch.argmax(output).item())
             
        return running_loss, posetives / (i + 1)


    def train(self):
        self.train_losses = []
        self.val_losses = []
        self.accuarcies = []
        for e in range(self.epochs):
            self.train_losses.append(self.trian_one_epock(e))
            val_loss, val_accu = self.validate_one_epoch()
            self.val_losses.append(val_loss)
            self.accuarcies.append(val_accu)
            print(f"Epoch: {e} traning_loss {self.train_losses[e]} validation loss: {self.val_losses[e]} Accu: {self.accuarcies[e]}")
        
        torch.save(self.m.state_dict(), self.model_path)
        return self.m, self.train_losses, self.val_losses, self.accuarcies

    def create_labels_tensors(self, labels_idx):
        if isinstance(labels_idx, int):
            labels_idx = torch.tensor([labels_idx])
        labels = torch.zeros(labels_idx.size()[0], 11)
        for i, l in enumerate(labels_idx):
            labels[i][l] = 1
        labels.squeeze()

        return labels[0].to(self.device) if labels.shape[0] == 1 else labels.to(self.device)

    def create_model_save_path(self):
        return os.path.join("saved_models", f"model_trained_{datetime.now().isoformat()}")


def main():
    tm = TrainModel()
    tm.validate_one_epoch()


if __name__ == "__main__":
    main()
    exit()
else:
    import pytest


def test_create_labels_tensors():
    import functools

    # Arrange
    labels_idx = torch.tensor([6, 5, 1, 10, 2, 3, 6, 10])
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    tm = TrainModel()
    # Act
    result = tm.create_labels_tensors(labels_idx)
    # Assert
    assert torch.equal(result, expected)


def test_create_labels_tensors_int_input():
    # Arrange
    labels_idx = 6
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    tm = TrainModel()
    # Act
    result = tm.create_labels_tensors(labels_idx)
    print(result)
    # Assert
    assert torch.equal(result, expected)
