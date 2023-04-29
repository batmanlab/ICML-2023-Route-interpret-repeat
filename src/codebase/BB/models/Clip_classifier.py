import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_features=1024, out_features=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.fc(x)
