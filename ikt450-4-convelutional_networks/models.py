import tensorflow as tf

## Convelutional layers
def convelutional_block(x, filters=32):
    x = tf.keras.layers.Activation("ReLU")(x)
    x = tf.keras.layers.Conv2D(
        kernel_size=(3, 3),
        filters=filters,
        padding="same",
        strides=1,
        data_format="channels_last"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def residual_block(x, filters=32):
    previus_block_activation = x # Save input for residual
    x = convelutional_block(x, filters) # Two convelutional layers
    x = convelutional_block(x, filters)
    # MaxPolling cutting the reselution in half
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    # Residual addtion
    ## Takes the Original imange, and takes every second value, 
    ## effectively cutting reselution in two
    residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same", data_format="channels_last")(
        previus_block_activation
    )
    # Adding the convelutional and pooled output with the residual
    return tf.keras.layers.add([x, residual])

x = residual_block(x, 32)

def make_model_from_exsample(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)
