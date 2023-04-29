import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from BB.models.BB_ResNet import ResNet
from BB.models.metanorm import MetadataNorm


def deactivate_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


class BB_ResNet101_metanorm(nn.Module):
    def __init__(self, args, dataset_size, kernel, train=True):
        super(BB_ResNet101_metanorm, self).__init__()
        self.dataset_size = dataset_size
        self.kernel = kernel

        bb = ResNet(
            dataset=args.dataset, pre_trained=args.pretrained, n_class=len(args.labels),
            model_choice=args.arch, layer=args.layer
        )
        if train:
            if args.finetune == "y":
                bb.load_state_dict(
                    torch.load(
                        os.path.join(args.checkpoints, args.dataset, 'BB', args.root_bb, args.arch, args.checkpoint_bb))
                )
                print("Finetune model loaded")
            if args.disable_batchnorm == "y":
                bb.apply(deactivate_batchnorm)
                print("Batchnorm disabled")

        self.backbone = torch.nn.Sequential(*(list(bb.base_model.children())[0:8]))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(bb.base_model.fc.weight.shape[1], bb.base_model.fc.weight.shape[1])
        self.fc2 = nn.Linear(bb.base_model.fc.weight.shape[1], bb.base_model.fc.weight.shape[1])
        self.fc3 = nn.Linear(bb.base_model.fc.weight.shape[1], len(args.labels))

        self.metadatanorm1 = MetadataNorm(
            cf_kernel=self.kernel, dataset_size=self.dataset_size, num_features=bb.base_model.fc.weight.shape[1]
        )

        self.metadatanorm2 = MetadataNorm(
            cf_kernel=self.kernel, dataset_size=self.dataset_size, num_features=bb.base_model.fc.weight.shape[1]
        )

        self.metadatanorm3 = MetadataNorm(
            cf_kernel=self.kernel, dataset_size=self.dataset_size, num_features=bb.base_model.fc.weight.shape[1]
        )

    def forward(self, x, cfs, get_phi=False):
        backbone = self.backbone(x)
        backbone = self.avg_pool(backbone).reshape(x.shape[0], -1)
        phi = F.relu(self.fc1(backbone))
        phi = self.metadatanorm1(phi, cfs)
        phi = F.relu(self.fc2(phi))
        phi = self.metadatanorm2(phi, cfs)
        y_hat = self.fc3(phi)
        if get_phi:
            return phi, y_hat
        else:
            return y_hat

    # def forward(self, x, get_phi=False):
    #     backbone = self.backbone(x)
    #     print(backbone.size())
    #     print(xxx)
    #     I = torch.eye(backbone.size(0)).to(device)
    #     xc = spurious_attr
    #     xc_T = torch.transpose(spurious_attr, 0, 1)
    #     phi_r_x = torch.mm(
    #         I - scale * torch.mm(xc, torch.mm(sigma, xc_T)),
    #         backbone.reshape(backbone.size(0), -1)
    #     ).reshape(-1, backbone.size(1), backbone.size(2), backbone.size(3))
    #     x = self.avg_pool(phi_r_x).reshape(x.shape[0], -1)
    #     x = self.fc(x)
    #     if get_phi:
    #         return phi_r_x
    #     else:
    #         return x
