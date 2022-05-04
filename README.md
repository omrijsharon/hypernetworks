# Hyper-Networks
Hyper-Networks implementation in Pytorch.

## How to use?
- import it:
```
from hypernetworks import HyperNetwork
```
- Create a parent model and a child model (the parent model outputs are the child model parameters/weights)
- Initialize a hyper-network made from a parent and child models:
```
model = HyperNetwork(parent_model, child_model)
```
- Prerare data that suits a hypernetwork and feed it to the hypernetwork model:
```
z = model(x, y)
```
- Notice the hyper-network gets 2 inputs:
```
child_weights = parent_model(x)
child_weights -> child_model
z = child_model(y)
```
