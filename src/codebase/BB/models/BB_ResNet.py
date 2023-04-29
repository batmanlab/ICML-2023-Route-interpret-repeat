import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        # init.constant_(m.bias.data, 0.0)


class ResNet(nn.Module):
    def __init__(self, dataset, pre_trained=True, n_class=200, model_choice="ResNet50", layer=None, as_residual=False):
        super(ResNet, self).__init__()
        self.feature_store = {}
        self.n_class = n_class
        self.as_residual = as_residual
        self.base_model = self._model_choice(pre_trained, model_choice)
        feat_dim = self.base_model.fc.weight.shape[1]

        if dataset == "cub" or dataset == "awa2" or dataset == "mnist":
            self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.base_model.fc = nn.Linear(in_features=feat_dim, out_features=n_class)
            self.base_model.fc.apply(weight_init_kaiming)

        if layer == "layer2":
            module = self.base_model.layer2
            module.register_forward_hook(self.save_activation(layer))

        elif layer == "layer3":
            module = self.base_model.layer3
            module.register_forward_hook(self.save_activation(layer))

        elif layer == "layer4":
            module = self.base_model.layer4
            module.register_forward_hook(self.save_activation(layer))

        elif layer == "adaptive":
            module = self.base_model.avgpool
            module.register_forward_hook(self.save_activation(layer))

    def save_activation(self, layer):
        def hook(module, input, output):
            self.feature_store[layer] = output

        return hook

    def forward(self, x):
        if self.as_residual:
            pass
        else:
            N = x.size(0)
            # assert x.size() == (N, 3, 448, 448)
            x = self.base_model(x)
            assert x.size() == (N, self.n_class)
            return x

    def save_gradient(self, grad):
        self.gradients = grad

    def generate_gradients(self, target, layer_name, bb_logit):
        activation = self.feature_store[layer_name]
        activation.register_hook(self.save_gradient)
        logit = bb_logit[:, target]
        logit.backward(torch.ones_like(logit), retain_graph=True)
        # gradients = grad(logit, activation, retain_graph=True)[0]
        # gradients = gradients.cpu().detach().numpy()
        gradients = self.gradients.cpu().detach()
        return gradients.squeeze()

    @staticmethod
    def _model_choice(pre_trained, model_choice):
        if model_choice == "ResNet50":
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == "ResNet101":
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == "ResNet152":
            return models.resnet152(pretrained=pre_trained)
