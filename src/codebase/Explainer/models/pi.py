import torch
import torch.nn as nn
from torch.nn import init


def weight_init_xavier(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.xavier_normal_(m.weight.data)
    elif class_names.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)


class Pi(nn.Module):
    def __init__(self, in_features=108, class_labels=200, params_init="xavier", data_heterogeneity=True):
        super(Pi, self).__init__()
        if data_heterogeneity:
            self.out = nn.Linear(in_features=in_features + class_labels, out_features=1)
        else:
            self.out = nn.Linear(in_features=class_labels, out_features=1)

        if params_init == "xavier":
            self.out.apply(weight_init_xavier)

    def forward(self, x, y_hat, data_heterogeneity_pi):
        y = y_hat.detach()
        # x_avg_pool = self.avgpool(x).reshape(-1, 2048 * 1 * 1)
        return self.out(torch.cat((x, y), dim=1) if data_heterogeneity_pi else y)
