import custemcallbacks
import keras
import dataloader
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import helper

def main():
    # set up helper 
    helper.static_name.model_name = "autoencoder"
    # stop
    image_size = (244, 244)
    data_test = dataloader.load_data_pair_no_label(
        path_x=dataloader.tests_paths["test"][0],
        path_y=dataloader.tests_paths["test"][1],
        image_size=image_size,
    )
    data_training = dataloader.load_data_pair_no_label(
        path_x=dataloader.paths_real["training"][0],
        path_y=dataloader.paths_real["training"][1],
        image_size=image_size,
        subset_procent=0.2,
    )
    data_validation = dataloader.load_data_pair_no_label(
        path_x=dataloader.paths_real["validation"][0],
        path_y=dataloader.paths_real["validation"][1],
        image_size=image_size,
        subset_procent=0.2,
    )
    data_evaluation = dataloader.load_data_pair_no_label(
        path_x=dataloader.paths_real["evaluation"][0],
        path_y=dataloader.paths_real["evaluation"][1],
        image_size=image_size,
        subset_procent=0.2,
    )
    # print lengths of datasets

    print(f"Length of data_test: {len(data_test)}")
    print(f"Length of data_training: {len(data_training)}")
    print(f"Length of data_validation: {len(data_validation)}")
    print(f"Length of data_evaluation: {len(data_evaluation)}")

    # Plot some images
    image: tf.data.Dataset = data_test.take(1)
    # subfigure
    for image_x, image_y in image:
        print(image_x.shape)
        print(image_y.shape)
        print(f"{np.max(image_x[0])} {np.min(image_x[0])}")
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image_x)
        plt.subplot(1, 2, 2)
        plt.imshow(image_y)
        plt.close()

    data_training = data_training.batch(8)
    data_validation = data_training.batch(8)
    # Time to build the model
    # Auto encoder
    for image_x, image_y in data_training.take(1):
        print(image_x.shape)
        print(image_y.shape)
        break

    input_shape = image_size + (3,)
    input_img = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    encoded = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = keras.Model(input_img, decoded)

    autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    autoencoder.summary()
    callback_list = [
        custemcallbacks.SaveBestModel(),
        custemcallbacks.LogMetricsToCSV(),
    ]

    # fit the model
    history = autoencoder.fit(
        data_training,
        epochs=10,
        batch_size=8,
        shuffle=True,
        validation_data=data_validation,
        callbacks=callback_list,
    )
    autoencoder.save("final_model") 

if __name__ == "__main__":
    main()
