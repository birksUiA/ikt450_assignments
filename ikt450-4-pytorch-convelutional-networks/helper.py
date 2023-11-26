import os
import torch
import matplotlib.pyplot as plt
import sklearn.metrics 

from datetime import datetime


classes =   {   0: "Class 1",
                1: "Class 2",
                2: "Class 3",
                3: "Class 4",
                4: "Class 5",
                5: "Class 6",
                6: "Class 7",
                7: "Class 8",
                8: "Class 9",
                9: "Class 10",
               10: "Class 11",
            }

class static_name:
    called_before = False 
    time_string = "ERROR"
    output_dir = "output"

    @staticmethod
    def _set_up():
        static_name.time_string = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")    
        if not os.path.exists(os.path.join(static_name.output_dir)):
            os.makedirs(os.path.join(static_name.output_dir))
        static_name.called_before = True

    @staticmethod
    def get_timed_file_name(prefix):
        if not isinstance(prefix, str): 
            raise TypeError("prefix not string!")

        if not static_name.called_before:
            static_name._set_up()

        return prefix + "_" + static_name.time_string

    @staticmethod
    def get_timed_file_path(prefix):
        if not isinstance(prefix, str): 
            raise TypeError("prefix not string!")

        if not static_name.called_before:
            static_name._set_up()

        return os.path.join(static_name.output_dir, static_name.get_timed_file_name(prefix))

def plot_imanges_with_lable(images, 
                yhat, # labels(actual or othervise)
                y=None, # actual labels, used when a tranined model is assesed
                rows=2, # rows of imanges
                cols=8, # cols of images
                figsize=(10, 4), # tuble of sizes of the figure. unit: inches
                save=True, # Should the figure be saved to disk?
                display=True # should the model be displayed when its created? 
                ): 
    fig, axs = plt.subplots(rows, cols, figsize=figsize,
                            sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat[0:]):
        ax.imshow(images[i].permute(1,2,0))
        title = f"{classes[torch.argmax(yhat[i].item())]}" if y == None else f"p: {classes[torch.argmax(yhat[i]).item()]}\na: {classes[torch.argmax(y[i]).item()]}"
        ax.set_title(title)

    if save:
        title = "images_"
        if not y == None:
            title += "and_predicted_vs_actual_labels" 
        else:
            title += "and_labels"

        plt.savefig(static_name.get_timed_file_path(title), dpi=100)

    if display:
        plt.show()

def plot_losses_during_training(train_losses, val_losses, save=True, display=True):
    """
    Plot the training and validation losses.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.

    Returns:
        None
    """
    # Set figure size
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(train_losses, label='Training Loss', linewidth=2, color='blue')

    # Plot validation loss
    plt.plot(val_losses, label='Validation Loss', linewidth=2, color='red')

    # Setting title, labels
    plt.title('Training and Validation Losses', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Displaying the grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjusting legend
    plt.legend(fontsize=12)

    # Save the figure
    if save: 
        plt.savefig(static_name.get_timed_file_path("loss_during_training"), dpi=100, bbox_inches='tight')

    if display:
        plt.show()

def plot_accuracies(accuracies, save=True, display=True):
    """
    """
    # Set figure size
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(accuracies, label='Training Loss', linewidth=2, color='blue')


    # Setting title, labels
    plt.title('Accuracies over Epochs', fontsize=16)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('accu', fontsize=14)

    # Displaying the grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjusting legend
    plt.legend(fontsize=12)

    # Save the figure
    if save: 
        plt.savefig(static_name.get_timed_file_path("accuracies"), dpi=100, bbox_inches='tight')
    # Show the figure
    if display:
        plt.show()


def plot_confustion_matrix(y, y_hat, display=True, save=False):
    cm =  sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_hat)
    cmd = sklearn.metrics.ConfusionMatrixDisplay(cm) 
    cmd.plot()

    if save:
        plt.savefig(static_name.get_timed_file_path("Confusion_matrix"), dpi=100)

    if display:
        plt.show()

def calculate_accuracy(y, y_hat):

    posetives = 0
    for i in range(len(y_hat)):
        if y[i] == y_hat[i]:
            posetives += 1

    return posetives / len(y_hat)

def main():
    print( static_name.get_timed_file_name("test"))
    print( static_name.get_timed_file_name("test2"))
    print( static_name.get_timed_file_path("test3"))
    print(static_name.output_dir) 

if __name__ == "__main__":
    main()
    exit()
else:
    import pytest

def test_plot_confutionmatrix():
    
    y = torch.randint(0, 5, (11, 1)) 
    y_hat = torch.randint(0, 5, (11, 1))
    print(F"y: {y}")
    print(F"y_hat: {y_hat}")
    plot_confustion_matrix(y=y, y_hat=y_hat)
    assert False



