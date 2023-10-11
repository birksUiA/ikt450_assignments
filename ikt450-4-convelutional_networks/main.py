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
import io

def main():
    # Set up
    print(f"\n\nGpu availible: {tf.test.is_gpu_available()}\n\n")
    print(f"\n\nGpu device name: {tf.test.gpu_device_name()}\n\n")
    epochs=100
    image_size = (244, 244)
    subset_procent = 0.4 

    traning_dataset, val_dataset, eval_dataset = dataloader.load_food_data(
        image_size, subset_procent
    )


    # define the model
    model = models.make_simple_convo_model(
        input_shape=image_size + (3,), num_classes=11
    )
    # Report on the defined model
    print(model.name)
    helper.static_name.model_name = model.name
    helper.plot_model(model)
    model.summary()

    helper.plot_images_from_set(dataset=traning_dataset, show=True, save=True)

    metrics = [
        "accuracy",
    ]

    initial_learning_rate = 0.1

    ## Compile the model - so that
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics,
    )

    # Fit the model
    # Define Callback functions

    callback_list = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=10, 
            verbose=True
        ),
        custemcallbacks.ConfusionMatrixCallback(val_dataset.rebatch(1)),
        custemcallbacks.SaveBestModel(),
        custemcallbacks.LogMetricsToCSV(),
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

    helper.plot_multiple_lines(
        xs=[history.history["accuracy"], history.history["val_accuracy"]],
        legneds=["accuracy", "val_accuracy"],
        title="Accuracy vs validation accuracy",
        ax_labels=("Epochs", "Acc"),
        save=True,
        show=False,
    )
    
    helper.plot_confustion_matrix(y=y_true, y_hat=y_pred, save=True, show=False)


if __name__ == "__main__":
    main()
    exit()
