import helper
import tensorflow as tf
from typing import Union
import os

paths_real = {
    # name: [path_in, path_out]
    "training": ["data/training/", "data/traning_timestamps/"],
    "validation": ["data/validation/", "data/validation_timestamps/"],
    "evaluation": ["data/evaluation/", "data/evaluation_timestamps/"],
}

tests_paths = {
    # name: [path_in, path_out]
    "test": ["data/test/", "data/test_timestamps/"],
}
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
        tf.keras.layers.RandomRotation(factor=(-0.15, 0.15)),
    ]
)
AUTOTUNE = tf.data.AUTOTUNE


def prepare_data(
    dataset: tf.data.Dataset,
    batch_size: int = 16,
    image_size: tuple[int, int] = (244, 244),
    augment: bool = False,
) -> tf.data.Dataset:
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, num_parallel_calls=AUTOTUNE)

    dataset = dataset.map(
        lambda x, y: (tf.keras.layers.Rescaling(1.0 / image_size[0])(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE
        )

    return dataset.prefetch(buffer_size=AUTOTUNE)


def load_food_data(
    image_size: tuple[int, int] = (244, 244), subset_procent: Union[None, float] = None
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    # Define paths
    traning_path = os.path.join("data", "training")
    validation_path = os.path.join("data", "validation")
    evaluation_path = os.path.join("data", "evaluation")

    # Load data
    traning_dataset = load_data(
        path=traning_path,
        image_size=image_size,
        subset_procent=subset_procent,
    )
    traning_dataset = prepare_data(traning_dataset, image_size=image_size, augment=True)
    val_dataset = load_data(
        path=validation_path, image_size=image_size, subset_procent=subset_procent
    )
    val_dataset = prepare_data(val_dataset, augment=True)

    eval_dataset = load_data(
        path=evaluation_path, image_size=image_size, subset_procent=subset_procent
    )
    eval_dataset = prepare_data(eval_dataset, batch_size=1, augment=True)

    return traning_dataset, val_dataset, eval_dataset


def load_data_pair_no_label(
    path_x: str,
    path_y: str,
    image_size: tuple[int, int],
    subset_procent: Union[None, float] = None,
) -> tf.data.Dataset:

    dataset_x: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        directory=path_x,
        shuffle=False,
        image_size=image_size,
        batch_size=None,
        label_mode=None,
    )

    dataset_x = dataset_x.map(
        lambda x: (tf.keras.layers.Rescaling(1.0 / 255)(x)),
        num_parallel_calls=AUTOTUNE,
    )
    dataset_y: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        directory=path_y,
        shuffle=False,
        image_size=image_size,
        batch_size=None,
        label_mode=None,
    )

    dataset_y = dataset_y.map(
        lambda x: (tf.keras.layers.Rescaling(1.0 / 255)(x)),
        num_parallel_calls=AUTOTUNE,
    )

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))

    dataset = dataset.shuffle(1000)
    if subset_procent is None:
        return dataset
    elif (
        isinstance(subset_procent, float) and subset_procent < 1 and subset_procent > 0
    ):
        return dataset.take(int(subset_procent * len(dataset)))
    else:
        raise ValueError(
            f"subset_procent must be a float between 0 and 1, not {subset_procent}"
        )


def load_data(
    path: str, image_size: tuple[int, int], subset_procent: Union[None, float] = None
) -> tf.data.Dataset:
    dataset: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        shuffle=True,
        image_size=image_size,
        batch_size=None,
        labels="inferred",
        label_mode="int",
    )

    if subset_procent is None:
        return dataset
    elif (
        isinstance(subset_procent, float) and subset_procent < 1 and subset_procent > 0
    ):
        return dataset.take(int(subset_procent * len(dataset)))
    else:
        raise ValueError(
            f"subset_procent must be a float between 0 and 1, not {subset_procent}"
        )


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

    counts_train = traning_dataset.reduce(
        initial_state=initial_state, reduce_func=count_class
    )
    initial_state = dict((i, 0) for i in range(num_classes))

    counts_val = val_dataset.reduce(
        initial_state=initial_state, reduce_func=count_class
    )
    initial_state = dict((i, 0) for i in range(num_classes))

    counts_eval = eval_dataset.reduce(
        initial_state=initial_state, reduce_func=count_class
    )

    print(
        f"Traning data class distribution: {[(k, v.numpy()) for k, v in counts_train.items()]}"
    )
    print(
        f"total traning images: {reduce(add, [v.numpy() for k, v in counts_train.items()])})"
    )
    print(
        f"validat data class distribution: {[(k, v.numpy()) for k, v in counts_val.items()]}"
    )
    print(
        f"total val images: {reduce(add, [v.numpy() for k, v in counts_val.items()])}"
    )
    print(
        f"evaluat data class distribution: {[(k, v.numpy()) for k, v in counts_eval.items()]}"
    )
    print(
        f"total eval images: {reduce(add, [v.numpy() for k, v in counts_eval.items()])}"
    )


if __name__ == "__main__":
    from functools import reduce
    from operator import add

    main()
