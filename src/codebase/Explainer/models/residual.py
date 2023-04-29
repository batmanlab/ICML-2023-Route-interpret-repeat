import torch.nn as nn
from torchvision import models


class Residual(nn.Module):
    def __init__(self, dataset, pre_trained=True, n_class=200, model_choice="ResNet50"):
        super(Residual, self).__init__()
        self.model_choice = model_choice
        self.feature_store = {}
        self.n_class = n_class
        self.feat_dim = self._model_choice(pre_trained, model_choice)
        self.avgpool = None
        self.fc = None
        self.dataset = dataset
        if (dataset == "cub" or dataset == "awa2" or dataset == "mnist") and (
                model_choice == "ResNet50" or model_choice == "ResNet101" or model_choice == "ResNet152"
        ):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=self.feat_dim, out_features=n_class)
        elif (dataset == "cub" or dataset == "mimic_cxr" or dataset == "awa2") and (
                model_choice == "ViT-B_16" or model_choice == "ViT-B_32_densenet"):
            self.fc = nn.Linear(in_features=self.feat_dim, out_features=n_class)
        elif dataset == "CIFAR10" and model_choice == "ResNet50":
            self.fc = nn.Linear(in_features=(self.feat_dim // 2), out_features=n_class)
        elif (
                dataset == "mimic_cxr" or dataset == "stanford_cxr" or dataset == "nih"
        ) and model_choice == "densenet121":
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.fc = nn.Linear(in_features=self.feat_dim, out_features=n_class)

    def forward(self, x):
        if self.model_choice == "ResNet50" or self.model_choice == "ResNet101" \
                or self.model_choice == "ResNet152" or self.model_choice == "densenet121":
            if self.dataset == "cub" or self.dataset == "mimic_cxr" or self.dataset == "stanford_cxr" or self.dataset == "nih" or self.dataset == "awa2" or self.dataset == "mnist":
                return self.fc(self.avgpool(x).reshape(-1, self.feat_dim * 1 * 1))
            elif self.dataset == "CIFAR10":
                return self.fc(x)
        elif self.model_choice == "ViT-B_16" or self.model_choice == "ViT-B_32_densenet":
            return self.fc(x.reshape(-1, self.feat_dim * 1 * 1))

    @staticmethod
    def _model_choice(pre_trained, model_choice):
        if model_choice == "ResNet50":
            return models.resnet50(pretrained=pre_trained).fc.weight.shape[1]
        elif model_choice == "ResNet101":
            return models.resnet101(pretrained=pre_trained).fc.weight.shape[1]
        elif model_choice == "ResNet152":
            return models.resnet152(pretrained=pre_trained).fc.weight.shape[1]
        elif model_choice == "densenet121":
            return 1024
        elif model_choice == "ViT-B_16":
            return 768
