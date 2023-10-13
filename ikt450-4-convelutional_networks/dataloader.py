import helper 
import tensorflow as tf
import os
import numpy as np 

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
    tf.keras.layers.RandomRotation(factor=(-0.15, 0.15)),
])
AUTOTUNE = tf.data.AUTOTUNE

def prepare_data(dataset, batch_size=16, image_size=(244, 244), augment=False):

    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(
        batch_size, 
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.map(
        lambda x, y: (tf.keras.layers.Rescaling(1./image_size[0])(x), y), 
        num_parallel_calls=AUTOTUNE
    )

    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y),
            num_parallel_calls=AUTOTUNE
        )

    return dataset.prefetch(buffer_size=AUTOTUNE)

    
def load_food_data(image_size=(244, 244), subset_procent=None):
    # Define paths
    traning_path = os.path.join("food-11", "training")
    validation_path = os.path.join("food-11", "validation")
    evaluation_path = os.path.join("food-11", "evaluation")


    # Load data
    traning_dataset = load_data(
        path=traning_path, 
        image_size=image_size, 
        subset_procent=subset_procent,
    )
    traning_dataset = prepare_data(
        traning_dataset,
        image_size=image_size,
        augment=True
    )
    val_dataset = load_data(
        path=validation_path, 
        image_size=image_size, 
        subset_procent=subset_procent
    )
    val_dataset = prepare_data(
        val_dataset,
        augment=True
    )    

    eval_dataset = load_data(
        path=evaluation_path, 
        image_size=image_size, 
        subset_procent=subset_procent
    )
    eval_dataset = prepare_data(
        eval_dataset,
        batch_size=1,
        augment=True
    )    

    return traning_dataset, val_dataset, eval_dataset

def load_data(path, image_size, subset_procent=None):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        shuffle=True,
        image_size=image_size,
        batch_size=None,
        labels="inferred",
        label_mode="int",
    )

    if subset_procent is None:
        return dataset
    elif isinstance(subset_procent, float) and subset_procent < 1 and subset_procent > 0:
        return dataset.take(int(subset_procent * len(dataset)))

def count_class(counts, batch, num_classes=11):
    print(type(batch))
    print(batch)
    labels = batch[1]
    for i in range(num_classes):
        cc = tf.cast(labels == i, tf.int32)
        counts[i] += tf.reduce_sum(cc)
    return counts

def main():
    image_size = (244, 244)
    subset_procent = None

    traning_dataset, val_dataset, eval_dataset = load_food_data(
        image_size, subset_procent
    )
    helper.static_name.model_name = "DataExploration"
    
    num_classes = 11
    initial_state = dict((i, 0) for i in range(num_classes))

    counts_train = traning_dataset.reduce(initial_state=initial_state,
                             reduce_func=count_class)
    initial_state = dict((i, 0) for i in range(num_classes))

    counts_val = val_dataset.reduce(initial_state=initial_state,
                             reduce_func=count_class)
    initial_state = dict((i, 0) for i in range(num_classes))

    counts_eval = eval_dataset.reduce(initial_state=initial_state,
                             reduce_func=count_class)
    
    
    print(f"Traning data class distribution: {[(k, v.numpy()) for k, v in counts_train.items()]}")
    print(f"total traning images: {reduce(add, [v.numpy() for k, v in counts_train.items()])})")
    print(f"validat data class distribution: {[(k, v.numpy()) for k, v in counts_val.items()]}")
    print(f"total traning images: {reduce(add, [v.numpy() for k, v in counts_train.items()])}")
    print(f"evaluat data class distribution: {[(k, v.numpy()) for k, v in counts_eval.items()]}")
    print(f"total traning images: {reduce(add, [v.numpy() for k, v in counts_train.items()])}")




if __name__ == "__main__": 
    from functools import reduce
    from operator import add
    main() 
