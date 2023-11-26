= Deep Feed Forward Neural Networks
In this assignment, a deep feed forward network should be implemented with a High Level API. 
The recomened API is PyTorch, though other APIs are allowed as well.
The dataset is a series of normal and abnormal ECGs. 
The goal is to classify the ECGs as normal or abnormal.

== Dataset
The dataset is a series of normal and abnormal ECGs.
Each enctry has two different leads, summing up to a potiential 150 features in total. 
Not every lead had 75 entries, in this case the result was padded or cut.
== Implementations

There where several(Many) attepmts to implement with PyTorch, but none of them worked.
While there where some arcitectures that simple where wrong, when a correct one was found, it would not train.
The paramenters would not update, between the epochs, even though the implementation followed the documentation. 

In the end a model was created with Keras, and the results where quite good.

== Results

#figure(
  grid(
    columns: (1fr, 1fr ),
    rows: (27%),
    gutter: 3pt,

    figure(kind: "subfigure", supplement: "Sub figure",
      image("../assets/3/loss.png", width: 100%),
      caption: [
        Loss over epochs
      ],
    ),
    figure(kind: "subfigure", supplement: "Sub figure",
      image("../assets/3/accuracy.png", width: 100%),
      caption: [
        Accuracies over epochs
      ],
    ),
  ),
  caption: [Loss and accuracy over epochs.]
)

As can been seen on the grapghs, the loss very quickly converges to a low value, while the accuracy for the traning goes up, but the accuracy for the test is a bit all over the place. 
It does however converge to a value of around 90%, witch is quite acceptble.
When evaluating the model on the test set, the following results are obtained:

```
Test loss:  0.44181203842163086
Test accuracy:  0.75
```
== Model recap

#figure(
  grid(
    columns: (2fr, 1fr ),
    gutter: 3pt,
[    
```
_______________________________________________
 Layer (type)                Output Shape              Param #   
===============================================
 dense (Dense)               (None, 1024)              77824     
 dropout (Dropout)           (None, 1024)              0         
 dense_1 (Dense)             (None, 512)               524800    
 dropout_1 (Dropout)         (None, 512)               0         
 dense_2 (Dense)             (None, 256)               131328    
 dropout_2 (Dropout)         (None, 256)               0         
 dense_3 (Dense)             (None, 64)                16448     
 dropout_3 (Dropout)         (None, 64)                0         
 dense_4 (Dense)             (None, 32)                2080      
 dropout_4 (Dropout)         (None, 32)                0         
 dense_5 (Dense)             (None, 1)                 33        
===============================================
```
    ],

  ),
    caption: [On the left is the model recaped, while on the right is the test loss and accuracy.]
)


