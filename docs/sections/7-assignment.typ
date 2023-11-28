
= Autoencoders
This assignments task is to remove a time stamp from an image using an auto encoder.

== Implementation 

The implementation was done with keras for the model and openCV for adding the text to the images.
The model is simple and a genneral case of an  auto encoders.
In this implementation it is inspired by @auto-impl.

An preview of how the timestamps images look are included in on @7-images, together with the results.

```rust
_________________________________________________________________
Layer (type)                    Output Shape              Param #
=================================================================
input_1 (InputLayer)            [(None, 244, 244, 3)]     0      
conv2d (Conv2D)                 (None, 244, 244, 32)      896    
max_pooling2d (MaxPooling2D)    (None, 122, 122, 32)      0      
conv2d_1 (Conv2D)               (None, 122, 122, 32)      9248   
max_pooling2d_1 (MaxPooling2D)  (None, 61, 61, 32)        0      
conv2d_2 (Conv2D)               (None, 61, 61, 32)        9248   
up_sampling2d (UpSampling2D)    (None, 122, 122, 32)      0      
conv2d_3 (Conv2D)               (None, 122, 122, 32)      9248   
up_sampling2d_1 (UpSampling2D)  (None, 244, 244, 32)      0      
conv2d_4 (Conv2D)               (None, 244, 244, 3)       867    
=================================================================
```

== Results

The simple autoencoder proved to be very effictent, yielding very good evaluation results: 
```
Loss: 0.5062946677207947, 
Accurasy: 0.9161678552627563
```

Supported by the quick converging accurassy and loss curces as seen on @7-graphs
#figure(
  grid(
    columns: (1fr, 1fr ),
    rows: (30%),
    gutter: 3pt,

    figure(kind: "subfigure", supplement: "Sub figure",
      image("../assets/7/Accuracy.png", height: 80%),
      caption: [
        Accuracy over epochs
      ],
    ),

    figure(kind: "subfigure", supplement: "Sub figure",
      image("../assets/7/loss.png", height: 80%),
      caption: [
        Loss over Epochs
      ],
    ),
  ),
  caption: [Accuracy and loss of the Auto encoder ]
)<7-graphs>

#figure(
  image("../assets/7/images.png", width: 70%),
  caption: [
    The original images, the images with time stamps, the predicted images.    
  ],
)<7-images>

