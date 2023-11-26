= KNN K Nearest Nairbours 

In this exercise a KNN algorithm was  implemented and tested on the pima-indians-diabetes.csv

== data

The dataset came as a csv file, and where split in a 80/20 split. 

Since KNN does not need to train or learn, the 80% training set was used as the neighbours of the 20% validation data. 

== Implementation

The KNN implementation chose the category based on the mean of neighbours. 
If the mean was above 0.5, the category was set to 1, else 0.
== Results

The algorithm was tested with K between 1 and 71, and the accuracy, recall, precision and F1 score are plotted on the graph below.

#figure(
  image("../assets/knn.png"),
  caption: ["KNN scores for different K values"]
)
As it can be seen, the accurassy slowly increases with K, but the recall and F1 decreases.
The precision does have a life of its own, peaking at the time the accuracy flattens out.

I don't know why. 

