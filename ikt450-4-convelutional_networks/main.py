"""

to solve all problems I need
 - a unified model import
 - Traning and verfication loss graph saved for each model
 - Accurassy, F1, Recall and precision graph for each model traning
 - Confusion mtrix for each model
 - to save each model for later use
"""

import os
import numpy as np
import tensorflow as tf
import helper


def main():
    # Set up helper
    helper.static_name.model_name = "TestModel"

    # Get the data as datasets
    evaluation_path = os.path.join("food-11", "validation")
    traning_path = os.path.join("food-11", "training")
    validation_path = os.path.join("food-11", "validation")
    image_size = (128, 128)

    # Split the data
    traning_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=traning_path,
        shuffle=True,
        image_size=image_size,
        labels="inferred",
        label_mode="int",
        batch_size=16,
    )
    traning_subset = traning_dataset.take(int(0.1 * len(traning_dataset)))
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=validation_path,
        shuffle=True,
        image_size=image_size,
        labels="inferred",
        label_mode="int",
        batch_size=16,
    )
    val_subset = val_dataset.take(int(0.1 * len(val_dataset)))

    eval_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=evaluation_path,
        shuffle=True,
        image_size=image_size,
        labels="inferred",
        label_mode="int",
        batch_size=1,
    )
    eval_subset = eval_dataset.take(int(0.1 * len(eval_dataset)))

    helper.plot_images_from_set(dataset=traning_dataset, show=False, save=False)

    # define the model
    inputs = tf.keras.Input(shape=image_size + (3,))

    ## Preproccesing
    x = tf.keras.layers.Rescaling(scale=1.0 / image_size[0])(inputs)

    ## Traning Randomness
    x = tf.keras.layers.RandomRotation(factor=(0.1, 0.1))(x)
    ## Convelutional layers
    x = tf.keras.layers.Conv2D(32, 3, padding="Same", strides=1, activation="ReLU")(x)

    x = tf.keras.layers.Flatten()(x)
    ## Dense layers
    output = tf.keras.layers.Dense(units=11, activation="softmax")(x)

    # create and Compile the model
    model = tf.keras.Model(inputs, output)

    helper.plot_model(model)

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
    ]
    ## Compile the model - so that
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=metrics,
    )

    # Fit the model
    epochs = 2
    # Define Callback functions

    class CustemCallbacks(tf.keras.callbacks.Callback):
        def __init__(self, test_data):
            self.test_dataset = test_data

        def on_epoch_end(self, epoch, log=None):
            # Evaluate the model
            ## First make prediction on the dataset.
            y_pred = model.predict(self.test_dataset)
            y_pred = np.argmax(y_pred, axis=1)
            ## Then extract to true labels
            y_true = self.test_dataset.map(lambda x, y: y)
            y_true = np.array(list(y_true.as_numpy_iterator()))
            y_true = np.transpose(y_true)
            y_true = np.squeeze(y_true)

            helper.plot_confustion_matrix(
                y=y_true,
                y_hat=y_pred,
                save=True,
                show=False,
                sub_dir="conv-matrixs",
                epoch=epoch,
            )
            accraccy = np.sum(np.equal(y_true, y_pred)) / len(y_pred)
            log["val_cal_accurracy"] = accraccy

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
           filepath=helper.static_name.get_timed_file_path("save_at_{epoch}.keras",
           save_best_only=True
            )
        ),
        CustemCallbacks(val_subset.rebatch(1)),
    ]

    history = model.fit(
        traning_subset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_subset,
    )

    # Evaluate the model
    ## First make prediction on the dataset.
    y_pred = model.predict(eval_subset)
    y_pred = np.argmax(y_pred, axis=1)
    ## Then extract to true labels
    y_true = eval_subset.map(lambda x, y: y)
    y_true = np.array(list(y_true.as_numpy_iterator()))
    y_true = np.transpose(y_true)
    y_true = np.squeeze(y_true)

    # Calculate Accuaracy

    helper.plot_losses_during_training(
        train_losses=history.history["loss"],
        val_losses=history.history["val_loss"],
        save=True,
        show=False,
    )

    helper.plot_accuracies(
        accuracies=history.history["categorical_accuracy"],
        accuracies_validation=history.history["val_categorical_accuracy"],
        save=True,
        show=False,
    )

    helper.plot_confustion_matrix(y=y_true, y_hat=y_pred, save=True, show=False)


if __name__ == "__main__":
    main()
    exit()
