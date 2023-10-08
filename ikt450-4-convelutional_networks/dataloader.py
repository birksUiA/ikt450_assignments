import tensorflow as tf
import os

def load_food_data(image_size=(244, 244), subset_procent=None):
    # Define paths
    traning_path = os.path.join("food-11", "training")
    validation_path = os.path.join("food-11", "validation")
    evaluation_path = os.path.join("food-11", "evaluation")


    # Load data
    traning_dataset = load_data(
        path=traning_path, 
        image_size=image_size, 
        subset_procent=subset_procent
    )

    val_dataset =  load_data(
        path=validation_path, 
        image_size=image_size, 
        subset_procent=subset_procent
    )    

    eval_dataset =  load_data(
        path=evaluation_path, 
        batch_size=1,
        image_size=image_size, 
        subset_procent=subset_procent
    )
    return traning_dataset, val_dataset, eval_dataset

def load_data(path, image_size, batch_size=32, subset_procent=None):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        shuffle=True,
        image_size=image_size,
        labels="inferred",
        label_mode="int",
        batch_size=32,
    )
    if subset_procent is None:
        return dataset
    elif isinstance(subset_procent, float) and subset_procent < 1 and subset_procent > 0:
        return dataset.take(int(subset_procent * len(dataset)))
