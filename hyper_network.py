import torch
from torch import nn
from copy import deepcopy


class HyperNetwork(nn.Module):
    def __init__(self, parent_model, child_model):
        super(HyperNetwork, self).__init__()
        self.child_num, self.child_shape = self.model_info(child_model)
        total = sum(self.child_num)
        self.child_model = child_model
        self.parent_model = nn.Sequential(parent_model, nn.Linear(list(parent_model.parameters())[-1].size(0), total))

    @staticmethod
    def model_info(model):
        shape = [tuple(param.size()) for param in list(model.parameters())]
        num = [torch.prod(torch.tensor(param_shape)).item() for param_shape in shape]
        return num, shape

    def _construct(self, child_params):
        batch_size = child_params.size(0)
        children_models = [deepcopy(self.child_model) for _ in range(batch_size)]
        split_params = self._split(child_params)
        reshaped_params = self._reshape(split_params)
        for child_num, child in enumerate(children_models):
            for layer_num, params in enumerate(child.parameters()):
                params.requires_grad_(False)
                params.copy_(reshaped_params[layer_num][child_num])
        return children_models

    def _split(self, child_params):
        return list(torch.split(child_params, self.child_num, dim=1))

    def _reshape(self, split_params: list):
        for i, params in enumerate(split_params):
            split_params[i] = params.reshape(params.size(0), *self.child_shape[i])
        return split_params

    def _beforward(self, x):
        children_params = self.parent_model(x)
        children_models = self._construct(children_params)
        return children_models

    def forward(self, x, y):
        children_models = self._beforward(x) # for every sample of x -> a tailored model is created.
        children_output = [child_model(y) for child_model in children_models]
        return torch.vstack(children_output)
