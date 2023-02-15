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
        """
        :param child_params: shape: (batch_size, total_num_of_params)
        :return: target models. list of models with length: batch_size
        """
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
        '''
        :param x: input to the parent model. shape: (batch_size, input_size)
        :param y: input to the child model. shape: (batch_size, input_size)
        :return: model output. shape: (batch_size, output_size)
        '''
        children_models = self._beforward(x) # for every sample of x -> a tailored model is created.
        children_output = [child_model(y) for child_model in children_models]
        return torch.vstack(children_output)


class HierarchicalHyperNetwork(nn.Module):
    def __init__(self, models_list: list):
        """
        :param models_list: [parent_model, child_model, grandchild_model, ...]
        """
        super(HierarchicalHyperNetwork, self).__init__()
        self.models = models_list
        self.target_num, self.target_shape, self.total = [], [], []
        for i in range(1, len(self.models)):
            # from grandchild to parent since only the youngest sibling has the correct number of parameters.
            target_num, target_shape = self.model_info(self.models[-i])
            total = sum(target_num)
            self.models[-i-1] = nn.Sequential(self.models[-i-1], nn.Linear(list(self.models[-i-1].parameters())[-i].size(0), total))
            self.target_num.append(target_num)
            self.target_shape.append(target_shape)
            self.total.append(total)
        target_num, target_shape = self.model_info(self.models[0])
        total = sum(target_num)
        self.target_num.append(target_num)
        self.target_shape.append(target_shape)
        self.total.append(total)
        self.target_num.reverse()
        self.target_shape.reverse()
        self.total.reverse()

    @staticmethod
    def model_info(model):
        shape = [tuple(param.size()) for param in list(model.parameters())]
        num = [torch.prod(torch.tensor(param_shape)).item() for param_shape in shape]
        return num, shape

    def _construct(self, target_model, target_params):
        split_params = self._split(target_params)
        reshaped_params = self._reshape(split_params)
        for layer_num, params in enumerate(target_model.parameters()):
            params.requires_grad_(False)
            params.copy_(reshaped_params[layer_num][0])
            params.requires_grad_(True)
        return target_model

    def _split(self, target_params, target_num):
        return list(torch.split(target_params, target_num, dim=1))

    def _reshape(self, split_params: list, shape):
        for i, params in enumerate(split_params):
            split_params[i] = params.reshape(params.size(0), *shape[i])
        return split_params

    def _beforward(self, hyper_model, inp, target_model):
        target_params = hyper_model(inp)
        self._construct(target_model, target_params)

    def forward(self, inps: list):
        '''
        :param list: list of tensor inputs to the parent model. [x, y, z...] each with shape (batch_size, input_size)
        :return: last model output. shape: (batch_size, output_size)
        '''
        for i in range(len(inps[0])):
            for j in range(len(self.models) - 1):
                self._beforward(self.models[j], inps[j][i].unsqueeze(0), self.models[j+1])

        # self.models = self._beforward(self.models[0][0], inps[0], self.models[1][0])
        # children_output = [child_model(y) for child_model in children_models]
        # return torch.vstack(children_output)