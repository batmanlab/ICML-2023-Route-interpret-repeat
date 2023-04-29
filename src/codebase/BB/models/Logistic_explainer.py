import torch


class Logistic_Regression(torch.nn.Module):
    def __init__(self, ip_size, op_size):
        super(Logistic_Regression, self).__init__()
        self.linear = torch.nn.Linear(in_features=ip_size, out_features=op_size, bias=True)

    def forward(self, x):
        return self.linear(x)
