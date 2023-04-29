import torch.nn as nn
# from models.model_resnet import resnet50, resnet101, resnet152
from torch.nn import init
# from models.model_resnet_ed import resnet50_ed, resnet101_ed, resnet152_ed
# from models.model_resnet_se import se_resnet50, se_resnet101, se_resnet152
from torchvision import models


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        # init.constant_(m.bias.data, 0.0)


class ResNet(nn.Module):
    def __init__(self, dataset, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        feat_dim = self.base_model.fc.weight.shape[1]
        if dataset == "cub":
            self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.base_model.fc = nn.Linear(in_features=feat_dim, out_features=n_class)
            self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x

    @staticmethod
    def _model_choice(pre_trained, model_choice):
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)
