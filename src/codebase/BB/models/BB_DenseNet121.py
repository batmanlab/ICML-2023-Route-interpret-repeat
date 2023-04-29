import pickle

import timm
import torch
from torch import nn


class DenseNet121(nn.Module):
    def __init__(self, args, layer="features_denseblock4"):
        super(DenseNet121, self).__init__()
        self.args = args
        self.feature_store = {}
        # define feature extractors
        self.backbone = timm.create_model(args.arch, pretrained=self.args.pretrained, features_only=True)

        # define pooling layers
        if self.args.pool1 == 'average':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.args.pool1 == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif self.args.pool1 == 'log-sum-exp':
            self.pool = self.logsumexp_pooling
        else:
            raise Exception('Invalid pooling layer type.')

        if layer == "features_denseblock3":
            module = self.backbone.features_transition3
            module.register_forward_hook(self.save_activation("features_transition3"))

        self.fc1 = nn.Linear(1024, args.N_labels)

    def save_activation(self, layer):
        def hook(module, input, output):
            self.feature_store[layer] = output

        return hook

    def forward(self, x):
        features = self.backbone(x)[-1]
        pooled_features = self.pool(features)
        logits = self.fc1(pooled_features.squeeze())

        return features, pooled_features, logits


if __name__ == "__main__":
    # "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/BB/lr_0.01_epochs_60/densenet121/pneumonia_pneumothorax"
    out_path = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/BB/lr_0.01_epochs_60/densenet121/pneumonia_pneumothorax/MIMIC_test_configs.pkl"
    args = pickle.load(open(out_path, "rb"))
    args.layer = "features_denseblock3"
    x = torch.rand(8, 3, 512, 512).cuda(args.gpu, non_blocking=True)
    model = DenseNet121(args, args.layer).cuda(args.gpu)
    print(model)
    # for i in range(11000):
    #     # print(model(x))
    #     print(model(x)[0].size())
    #     print(model.feature_store["features_transition3"].size())
    #     print(model.fc1.weight.shape[1])
