= Neural Networks

In this assignment two multiplayed perseptrons where to be developed. 
One with and one without a higher level libary to do the work for us. 

== Custom implementation

The implemetnation was very simple, where the layer class had a forward method that simply: 

```
return self.activate(np.matmul(self.w, x))
```
Ofcuse a represent method was also implented, so a small model can be seen here:
```output
0: Layer(input_size =   5, output_size =   4, activation=relu)
1: Layer(input_size =   4, output_size =   4, activation=relu)
2: Layer(input_size =   4, output_size =   1, activation=sigmoid)
```
The result on running this on the dataset can be seen here:
```
L2 loss:  [0.35795455]
accuracy:  0.6420454545454546
precision:  0.6420454545454546
recall:  1.0
F1:  0.7820069204152249
```
Where it seems that the random initiation of the weights where a little better at predicting then random.
== Higher level library

The implementation with a higher level library was down with PyTorch.
```Output
SimpleNet(
    (fc1): Linear(in_features=5, out_features=5, bias=True)
    (relu): ReLU()
    (sigmoid): Sigmoid()
    (fc2): Linear(in_features=5, out_features=10, bias=True)
    (fc3): Linear(in_features=10, out_features=1, bias=True)
)
```
Trained a hundred epochs, this seem to have over fit, as we get perfect metrics back. 

```
accuracy:  1.0
precision:  1.0
recall:  1.0
F1:  1.0
```

The loss over the epochs can be seen on @2-loss-curve

#figure(
  image("../assets/ass2-loss-over-epoch.png", width: 80%),
  caption: ["Loss over 100 epochs"]
)<2-loss-curve>
