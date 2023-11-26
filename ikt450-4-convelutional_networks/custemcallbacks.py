import tensorflow as tf
import numpy as np
import helper
import time


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_dataset = test_data

    def on_epoch_end(self, epoch, log=None):
        if (epoch + 1) % 10 == 0:
            # Evaluate the model
            ## First make prediction on the dataset.
            t0 = time.time()
            y_pred = self.model.predict(self.test_dataset)
            y_pred = np.argmax(y_pred, axis=1)
            ## Then extract to true labels
            y_true = self.test_dataset.map(lambda x, y: y)
            y_true = np.array(list(y_true.as_numpy_iterator()))
            y_true = np.transpose(y_true)
            y_true = np.squeeze(y_true)

            t1 = time.time()
            helper.plot_confustion_matrix(
                y=y_true,
                y_hat=y_pred,
                save=True,
                show=False,
                sub_dir="conv-matrixs",
                epoch=epoch,
            )
            print(f"\nTime to predict:          {t1-t0}\n")
            print(f"\n{log}\n")


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, best_loss=float("inf")):
        self.best_loss = best_loss

    def on_epoch_end(self, epoch, log=None):
        if log["val_loss"] < self.best_loss:
            self.best_loss = log["val_loss"]
            self.model.save(
                helper.static_name.get_timed_file_path(
                    sub_dir="saved_model", file_name=f"model_best_loss_at_epoch_{epoch}"
                )
            )
            print(f"model saved at epoch {epoch}")


class LogMetricsToCSV(tf.keras.callbacks.Callback):
    def __init__(self):
        self.path = helper.static_name.get_timed_file_path(file_name="metrics.csv")
        with open(self.path, "wt") as f:
            f.write("loss,val_loss,accuracy,vel_accuracy\n")

    def on_epoch_end(self, epoch, log=None):
        with open(self.path, "at") as f:
            f.write(
                f"{log['loss']},{log['val_loss']},{log['accuracy']},{log['val_accuracy']}\n"
            )
