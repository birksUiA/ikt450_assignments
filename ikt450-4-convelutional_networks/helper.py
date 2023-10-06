from sys import prefix
import sklearn.metrics
import matplotlib.pyplot as plt
from datetime import datetime
import os
import tensorflow as tf

classes = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles/Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable/Fruit",
]
classes_dict = {
    0: classes[0],
    1: classes[1],
    2: classes[2],
    3: classes[3],
    4: classes[4],
    5: classes[5],
    6: classes[6],
    7: classes[7],
    8: classes[8],
    9: classes[9],
    10: classes[10],
}


class static_name:
    called_before = False
    time_string = "ERROR"
    output_dir = "ERROR"
    model_name = "ERROR"
    dpi = 320

    def __init__(self, model_name="unnamed_model"):
        static_name._set_up(model_name)

    @staticmethod
    def _set_up(model_name="unnamed_model"):
        static_name.time_string = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        if model_name == "ERROR":
            static_name.model_name = model_name
        static_name.output_dir = os.path.join(
            "output",
            static_name.model_name,
            static_name.time_string,
        )
        if not os.path.exists(os.path.join(static_name.output_dir)):
            os.makedirs(
                os.path.join(
                    static_name.output_dir,
                )
            )
        static_name.called_before = True

    @staticmethod
    def get_timed_file_name(prefix):
        if not isinstance(prefix, str):
            raise TypeError("prefix not string!")

        if not static_name.called_before:
            static_name._set_up()

        return prefix + "_" + static_name.time_string

    @staticmethod
    def get_timed_file_path(file_name, sub_dir=None):
        if not isinstance(file_name, str):
            raise TypeError("prefix not string!")

        if not static_name.called_before:
            static_name._set_up()

        if sub_dir is None:
            return os.path.join(
                static_name.output_dir,
                static_name.get_timed_file_name(file_name),
            )
        elif isinstance(sub_dir, str) or isinstance(sub_dir, os.PathLike):
            path = os.path.join(
                static_name.output_dir,
                sub_dir,
            )
            if not os.path.exists(path=path):
                os.makedirs(path)
            return os.path.join(path, file_name)
        else:
            raise RuntimeError("Sub_path was not string or os.PathLike")


def plot_images_from_set(
    dataset,
    number_of_images=9,
    image_grid=(3, 3),
    figsize=(10, 10),
    save=False,
    show=True,
):
    plt.figure(figsize=figsize)
    for images, labels in dataset.take(1):
        for i in range(number_of_images):
            ax = plt.subplot(image_grid[0], image_grid[1], i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classes_dict[int(labels[i])])
            plt.axis("off")

    if save:
        plt.savefig(
            static_name.get_timed_file_path("images_with_true_labels"),
            dpi=static_name.dpi,
        )
    if show:
        plt.show()


def plot_model(model, show_shapes=True):
    tf.keras.utils.plot_model(
        model=model,
        show_shapes=True,
        to_file=static_name.get_timed_file_path("model_plot") + ".png",
    )


def plot_losses_during_training(train_losses, val_losses, save=True, show=True):
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
    plt.plot(train_losses, label="Training Loss", linewidth=2, color="blue")

    # Plot validation loss
    plt.plot(val_losses, label="Validation Loss", linewidth=2, color="red")

    # Setting title, labels
    plt.title("Training and Validation Losses", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # Displaying the grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjusting legend
    plt.legend(fontsize=12)

    # Save the figure
    if save:
        plt.savefig(
            static_name.get_timed_file_path("loss_during_training"),
            dpi=static_name.dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()


def plot_accuracies(accuracies, accuracies_validation=None, save=True, show=True):
    """ """
    # Set figure size
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(accuracies, label="accuracies", linewidth=2, color="blue")

    if accuracies_validation != None:
        plt.plot(
            accuracies_validation,
            label="Validation Accuracies",
            linewidth=2,
            color="green",
        )

    # Setting title, labels
    plt.title("Accuracies over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracies", fontsize=14)

    # Displaying the grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjusting legend
    plt.legend(fontsize=12)

    # Save the figure
    if save:
        plt.savefig(
            static_name.get_timed_file_path("accuracies"),
            dpi=static_name.dpi,
            bbox_inches="tight",
        )
    # Show the figure
    if show:
        plt.show()


def plot_confustion_matrix(y, y_hat, show=True, save=False, sub_dir=None, epoch=None):
    # cm = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_hat)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    cmd = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y,
        y_pred=y_hat,
        display_labels=classes,
        xticks_rotation="vertical",
        ax=ax,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confustion Matrix")

    if save:
        if not sub_dir is None:
            if not isinstance(epoch,int):
                raise RuntimeError(f"Epochs needed for subdir conv epoch: {epoch} type: {type(epoch)}")
            plt.savefig(
                static_name.get_timed_file_path(
                    file_name=f"Confusion_matrix_{epoch}", 
                    sub_dir=sub_dir
                ),
                dpi=static_name.dpi,
            )
        else:
            plt.savefig(
                static_name.get_timed_file_path("Confusion_matrix"), dpi=static_name.dpi
            )

    if show:
        plt.show()
