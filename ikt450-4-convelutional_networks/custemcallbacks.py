import tensorflow as tf
import numpy as np
import helper
class CustemCallbacks(tf.keras.callbacks.Callback):
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
