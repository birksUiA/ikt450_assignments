= Object detection
In this task, object detection models are to be explored with a small balloon dataset. 
Both object detection and segmentation should be explored.

== Implementation
The coice of model and libaryie for this fell on ultralytics YOLOv8 model. 
This made the implementation very simple, and the model used is summericed in @5-model.

== The dataset
The dataset came pre annotated for segmentation, then the dataset was converted into to correct format via roboflow. 

#set page(flipped: true)
== Model <5-model>
The model used was model YOLOv8 @yolov8
Here we can see that the models are identical, expept from the head
#figure(
  grid(
    columns: (1fr,1fr ),
    rows: (auto),
    gutter: 0.1em,
align(left, 
```
   params  module        arguments                     
 0    464  conv.Conv     [3, 16, 3, 2]                 
 1   4672  conv.Conv     [16, 32, 3, 2]                
 2   7360  block.C2f     [32, 32, 1, True]             
 3  18560  conv.Conv     [32, 64, 3, 2]                
 4  49664  block.C2f     [64, 64, 2, True]             
 5  73984  conv.Conv     [64, 128, 3, 2]               
 6 197632  block.C2f     [128, 128, 2, True]           
 7 295424  conv.Conv     [128, 256, 3, 2]              
 8 460288  block.C2f     [256, 256, 1, True]           
 9 164608  block.SPPF    [256, 256, 5]                 
10      0  Upsample*     [None, 2, 'near']          
11      0  conv.Concat   [1]                           
12 148224  block.C2f     [384, 128, 1]                 
13      0  Upsample*     [None, 2, 'near']          
14      0  conv.Concat   [1]                           
15  37248  block.C2f     [192, 64, 1]                  
16  36992  conv.Conv     [64, 64, 3, 2]                
17      0  conv.Concat   [1]                           
18 123648  block.C2f     [192, 128, 1]                 
19 147712  conv.Conv     [128, 128, 3, 2]              
20      0  conv.Concat   [1]                           
21 493056  block.C2f     [384, 256, 1]                 
22 751507  head.Detect   [1, [64, 128, 256]] 
```),
align(left, ```
  params  module         arguments                     
     464  conv.Conv      [3, 16, 3, 2]                 
    4672  conv.Conv      [16, 32, 3, 2]                
    7360  block.C2f      [32, 32, 1, True]             
   18560  conv.Conv      [32, 64, 3, 2]                
   49664  block.C2f      [64, 64, 2, True]             
   73984  conv.Conv      [64, 128, 3, 2]               
  197632  block.C2f      [128, 128, 2, True]           
  295424  conv.Conv      [128, 256, 3, 2]              
  460288  block.C2f      [256, 256, 1, True]           
  164608  block.SPPF     [256, 256, 5]                 
       0  Upsample       [None, 2, 'nearest']          
       0  conv.Concat    [1]                           
  148224  block.C2f      [384, 128, 1]                 
       0  Upsample       [None, 2, 'nearest']          
       0  conv.Concat    [1]                           
   37248  block.C2f      [192, 64, 1]                  
   36992  conv.Conv      [64, 64, 3, 2]                
       0  conv.Concat    [1]                           
  123648  block.C2f      [192, 128, 1]                 
  147712  conv.Conv      [128, 128, 3, 2]              
       0  conv.Concat    [1]                           
  493056  block.C2f      [384, 256, 1]                 
 1004275  head.Segment   [1, 32, 64, [64, 128, 256]]
```),
  ),
  caption: [On the left the dection model is seen, while on the rihgt the segmentatio model is seen.]
)
Where \* indicates that the modules came from `torch.nn.modules.upsampling`.
The rest all come from `ultralytics.nn.modules`
#set page(flipped: false)
#pagebreak()
== Detection results 
#figure(
  grid(
    columns: (1fr, 1fr ),
    rows: (30%),
    gutter: 3pt,

    figure(
      image("../assets/5/det/Detection mAPs.png",width: 100%, height: 80%),
      caption: [
        Accuracies
      ],
    ),
    figure(
      image("../assets/5/det/Detection Losses.png",width: 100%, height: 80%),
      caption: [
        Losses 
      ],
    ),
  ),
  caption: [Accuracies and loss during training]
)
Here it seen thta the model does not quite converge to attuaclly predict the boxes. 
This can allso bee seen on the predicted images on @7-det-image
#figure(
  image("../assets/5/det/val_batch0_pred.jpg", width: 70%),
  caption: [
    Last Validation prediction
  ],
) <7-det-image>

#pagebreak()
== Segmentation results
The segmentation results where better in general, as can be seen on the converging accuracies and the continues declining loss rates.
Here is also a clue to why the dections results where bad. 
The data set might not have been correctly formatted for box detection. 
#figure(
  grid(
    columns: (1fr, 1fr ),
    rows: (30%),
    gutter: 3pt,

    figure(
      image("../assets/5/seg/Segmentation mAPs.png", height: 80%),
      caption: [
        Accuracies
      ],
    ),
    figure(
      image("../assets/5/seg/Segmentataion Losses.png", height: 80%),
      caption: [
        Losses 
      ],
    ),
  ),
  caption: [Confusion matrix and images with predictions of the vgg]
)

#figure(
  image("../assets/5/seg/val_batch0_pred.jpg", width: 70%),
  caption: [
    Last Validation prediction
  ],
)
