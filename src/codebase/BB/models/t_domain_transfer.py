import torch

from BB.models.BB_DenseNet121 import DenseNet121


class T_Domain_Transfer(torch.nn.Module):
    def __init__(self, args, chk_pt_path_bb_mimic, ip_size, op_size):
        super(T_Domain_Transfer, self).__init__()
        self.ip_size = ip_size
        bb_mimic = DenseNet121(args, layer=args.layer).cuda()
        bb_mimic.load_state_dict(torch.load(chk_pt_path_bb_mimic)['state_dict'])
        self.features = bb_mimic.backbone
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=ip_size, out_features=op_size, bias=True)

    def forward(self, x):
        features = self.features(x)[-1]
        return self.linear(self.flatten(features))

