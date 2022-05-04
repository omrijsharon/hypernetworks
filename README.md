# Hyper-Networks
Hyper-Networks implementation in Pytorch.

## Requirements
Tested on python 3.8 and pytorch 1.10.

## How to use?
- import it:
```
from hypernetworks import HyperNetwork
```
- Create a parent model and a child model (the parent model outputs are the child model parameters/weights)
- Initialize a hyper-network module with a parent model and a child model:
```
model = HyperNetwork(parent_model, child_model)
```
- Prepare data that suits a hypernetwork and feed it to the hypernetwork model:
```
z = model(x, y)
```
- Notice the hyper-network gets 2 inputs:
```
child_weights = parent_model(x)
child_model <- child_weights
z = child_model(y)
```
## A code example
This code example uses torch_x (torch extensions package):
```
    batch_size = 6
    num_features = 64
    parent = MLP([num_features, 16, 10], nn.ReLU(), nn.Softmax(dim=1))
    child = MLP([10, 32, 1], nn.ReLU(), nn.Sigmoid())
    x = torch.randn(batch_size, num_features)
    y = torch.randn(3, 10)
    model = HyperNetwork(parent, child)
    model(x, y)
```
