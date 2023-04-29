import torch


class PCBM(torch.nn.Module):
    def __init__(self, ip_size, op_size):
        super(PCBM, self).__init__()
        self.model = torch.nn.Linear(in_features=ip_size, out_features=op_size, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x
