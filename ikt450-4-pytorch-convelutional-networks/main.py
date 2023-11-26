
from matplotlib.pyplot import functools, plot
from convo_model import ConvoModel
from para_model import ParaModel
from dataparser import get_sub_set, load_images
import helper
from helper import plot_imanges_with_lable
from train_model import TrainModel
import torch.utils.data

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    traning_dataset = get_sub_set(load_images(path="food-11/training/",
                                              resize_to=128,
                                              center_crop_to=128), 
                                  0.2)
    validation_dataset = get_sub_set(load_images(path="food-11/validation/",
                                                 resize_to=128,
                                                 center_crop_to=128), 
                                     0.2)

    untrianed_model = ConvoModel(image_height=128, image_width=128)
    untrianed_model.to(device)
    print(untrianed_model)

    loss_fn = torch.nn.MSELoss()

    tm = TrainModel(model=untrianed_model,
                    epochs=100,
                    loss_function=loss_fn,
                    traning_dataset=traning_dataset, 
                    validition_dataset=validation_dataset)


    tranined_model, train_losses, validation_losses, accuracies = tm.train()


    helper.plot_losses_during_training(train_losses, validation_losses, display=False, save=True)
    helper.plot_accuracies(accuracies, display=False, save=True)
    helper.plot_confustion_matrix(tm.y, tm.y_hat, display=False, save=True)

    # See prediction of model 
    eval_set = get_sub_set(load_images("./food-11/evaluation/",
                                       resize_to=128, 
                                       center_crop_to=128),
                           0.2)
    eva_dataloader = torch.utils.data.DataLoader(eval_set, shuffle=True, batch_size=16)
    
    images, labels = next(iter(eva_dataloader))
    yhat = tranined_model(images.to(device))

    plot_imanges_with_lable(images=images, yhat=yhat, y=tm.create_labels_tensors(labels), save=True)

if __name__ == "__main__":
    main()
    exit()
else:
    import pytest


def test_getting_prediction():
    expected = torch.tensor([3, 3, 3])
    data = torch.tensor([[0.1, 0.2, 0.67, 0.9, 0.1],
                         [0.1, 0.2, 0.67, 0.9, 0.1],
                         [0.1, 0.2, 0.67, 0.9, 0.1],
    ])
    func = lambda output: output.max()[1]    

    result:torch.Tensor() = map(func, data)
    print(result)

    assert torch.equal(expected, result) 
