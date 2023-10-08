import tensorflow as tf
import numpy as np
import helper


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_dataset = test_data

    def on_epoch_end(self, epoch, log=None):
        # Evaluate the model
        ## First make prediction on the dataset.
        y_pred = self.model.predict(self.test_dataset)
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
        print(f"{log}")
            
class SaveBestModel(tf.keras.callbacks.Callback):

    def __init__(self, best_loss=float('inf')):
        self.best_loss=best_loss  

    def on_epoch_end(self, epoch, log=None):
        if log["val_loss"] < self.best_loss:
            self.best_loss = log["val_loss"]
            self.model.save(
                helper.static_name.get_timed_file_path(
                    sub_dir="saved_model",
                    file_name=f"model_best_loss_at_epoch_{epoch}"
                )
            )
            print(f"model saved at epoch {epoch}")
