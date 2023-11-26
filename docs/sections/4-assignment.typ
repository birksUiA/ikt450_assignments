= Convolutional Neural Network

This assignment deals with the development and evaluation of a convolutional neural network (CNN) with a high level library. 

== Introduction

The implementation of this assignment took an unreasonable about of time and attempt. 
First PyTorch was used to create small toy models, though even at the scale of single convolutional layers, the models were not able to start due to various errors.
With everything from memory errors to implementation mistakes, pytorch was eventually abandoned in favor of Keras and Tensorflow.
The keras and Tensorflow solution where not without their own problems, and both small and large models were created and tested, all the way to mimicking the VGG16 and ResNet modeles.
Some of the models can bee soon on @4-models.
While the best results can be seen under @4-vgg-result and @4-res-result


== VGG16 results  <4-vgg-result>

#figure(
  image("../assets/4/vgg/Accuracy_vs_validation_accuracy.png", width: 70%),
  caption: [
    Accuracies over epochs for the vgg like model
  ],
)

#figure(
  grid(
    columns: (1fr, 1fr ),
    rows: (30%),
    gutter: 3pt,

    figure(
      image("../assets/4/vgg/Confusion_matrix.png", height: 80%),
      caption: [
        Confusion matrix
      ],
    ),
    figure(
      image("../assets/4/vgg/images_with_true_labels.png", height: 80%),
      caption: [
        Predictions with their images
      ],
    ),
  ),
  caption: [Confusion matrix and images with predictions of the vgg]
)
#pagebreak()
== ResNet results <4-res-result>

#figure(
  image("../assets/4/residual/Accuracy_vs_validation_accuracy.png", width: 70%),
  caption: [
    Accuracies over epochs for the lager residual model
  ],
)

#figure(
  grid(
    columns: (1fr, 1fr ),
    rows: (30%),
    gutter: 3pt,

    figure(
      image("../assets/4/residual/Confusion_matrix.png", height: 80%),
      caption: [
        Confustion matrix
      ],
    ),
    figure(
      image("../assets/4/residual/images_with_true_labels.png", height: 80%),
      caption: [
        Predictions with their images
      ],
    ),
  ),
  caption: [Confusion matrix and images with predictions of the larger custom residual model]
)

== Refrelction

On both models, it does not get that presice overall. 
This is very clear from both confusion matrixes that has not collapsed in to the diagonal, and the fact that it calls pasta for rice. 

The vgg is better then res. It should be the other way around. 
It is not known why.

#set page(paper: "a3")
== convolutional models
#figure(
  grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    rows: (auto),
    gutter: 3pt,
    
    figure(
      image("../assets/4/models/simple_residual.png", height: 80%),
      caption: [
        simple custom residual model 
      ],
    )
  ,
    figure(
      image("../assets/4/models/bigger_residual.png", height: 80%),
      caption: [
        Larger custom residual model 
      ],
    ),

    figure(
      image("../assets/4/models/vgg.png", height: 70%),
      caption: [
        custem implemetation of vgg
      ],
    ),

    figure(
      image("../assets/4/models/pre_trained.png", height: 80%),
      caption: [
        Use of a pre-defined and trained ResNet
      ],
    ),
  ),
  caption: [A small subset of models used and tested in this assignment]
) <4-models>
#pagebreak()
#set page(paper: "a4")
