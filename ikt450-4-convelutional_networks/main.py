"""

to solve all problems I need
 - a unified model import
 - Traning and verfication loss graph saved for each model
 - Accurassy, F1, Recall and precision graph for each model traning
 - Confusion mtrix for each model
 - to save each model for later use
"""

import numpy as np
import tensorflow as tf
import helper
import dataloader
import models
import custemcallbacks


def main():
    # Set up helper
    image_size = (244, 244)
    subset_procent = 0.1

    traning_dataset, val_dataset, eval_dataset = dataloader.load_food_data(
        image_size, subset_procent
    )

    helper.plot_images_from_set(dataset=traning_dataset, show=False, save=False)

    # define the model
    model = models.make_residual_model(
        input_shape=image_size + (3,), num_classes=11
    )
    print(model.name)
    helper.static_name.model_name = model.name
    helper.plot_model(model)

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
    ]

    initial_learning_rate = 0.1

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100, decay_rate=0.05
    )
    ## Compile the model - so that
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=metrics,
    )

    # Fit the model
    epochs = 1
    # Define Callback functions

    callback_list = [
        custemcallbacks.ConfusionMatrixCallback(val_dataset.rebatch(1)),
        custemcallbacks.SaveBestModel(),
    ]

    history = model.fit(
        traning_dataset,
        epochs=epochs,
        callbacks=callback_list,
        validation_data=val_dataset,
    )

    # Evaluate the model
    ## First make prediction on the dataset.
    y_pred = model.predict(eval_dataset)
    y_pred = np.argmax(y_pred, axis=1)
    ## Then extract to true labels
    y_true = eval_dataset.rebatch(1).map(lambda x, y: y)
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
