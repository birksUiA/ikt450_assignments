import numpy as np
import os
from data_loader import load_ecg_data, split_data

import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam


def main():
    # set random seed
    np.random.seed(42)
    # check if output folder exists if it does not exist create it else make
    output = "output"
    if not os.path.exists(output):
        os.makedirs(output)
    else:
        print("Output folder already exists")
        # add time stamp to output folder
        import time

        output = output + "_" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(output)

    # get device
    # Load data

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ecg")
    data = load_ecg_data(data_path)

    # Suffel the data
    shuffeled_indicers = np.random.permutation(len(data[0]))
    data[0] = data[0][shuffeled_indicers]
    data[1] = data[1][shuffeled_indicers]

    # split the data into train validation and test
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data[0], data[1])
    # print distribution of classes
    print("Train: ", np.unique(y_train, return_counts=True))
    print("Validation: ", np.unique(y_val, return_counts=True))
    print("Test: ", np.unique(y_test, return_counts=True))

    input_dim = x_train.shape[1]

    # create model
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(1024, activation="relu"),
            Dropout(0.5),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    # compile model
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
    )
    history = model.fit(
        x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val)
    )

    # plot history
    plt.figure(0)
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(output, "loss.png"))
    # plot acc
    plt.figure(1)
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output, "accuracy.png"))

    #
    # evaluate model
    loss, acc = model.evaluate(x_test, y_test)
    print("Test loss: ", loss)
    print("Test accuracy: ", acc)
    model.summary()
    # save model
    model.save(os.path.join(output, "model.h5"))


if __name__ == "__main__":
    main()
