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
import models
import multiprocessing
import custemcallbacks 

def main():
    # Set up helper
    helper.static_name.model_name = "vgg_like_model"

    # Get the data as datasets
    evaluation_path = os.path.join("food-11", "evaluation")
    traning_path = os.path.join("food-11", "training")
    validation_path = os.path.join("food-11", "validation")

    image_size = (244, 244)
    sub_set_procent = 0.3
    # Split the data
    traning_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=traning_path,
        shuffle=True,
        image_size=image_size,
        labels="inferred",
        label_mode="int",
        batch_size=32,
    )
    traning_subset = traning_dataset.take(int(sub_set_procent * len(traning_dataset)))
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=validation_path,
        shuffle=True,
        image_size=image_size,
        labels="inferred",
        label_mode="int",
        batch_size=32,
    )
    val_subset = val_dataset.take(int(sub_set_procent * len(val_dataset)))

    eval_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=evaluation_path,
        shuffle=True,
        image_size=image_size,
        labels="inferred",
        label_mode="int",
        batch_size=1,
    )
    eval_subset = eval_dataset.take(int(sub_set_procent * len(eval_dataset)))

    helper.plot_images_from_set(dataset=traning_dataset, show=False, save=False)

    # define the model
    model = models.make_vgg_like_convo_model(input_shape=image_size + (3,), num_classes=11)

    helper.plot_model(model)

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
    ]

    initial_learning_rate = 0.1

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.05
    )

    ## Compile the model - so that
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, 
            momentum=0.9
        ),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=metrics,
    )

    # Fit the model
    epochs = 1000
    # Define Callback functions

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
           filepath=helper.static_name.get_timed_file_path("save_at_{epoch}.keras"),
           save_best_only=True
        ),
        custemcallbacks.CustemCallbacks(val_subset.rebatch(1)),
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
