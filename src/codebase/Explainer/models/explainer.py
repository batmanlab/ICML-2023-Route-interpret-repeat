import torch
import torch.nn as nn

from Explainer.models.entropy_layer import EntropyLinear


class Explainer(nn.Module):
    def __init__(
            self,
            n_concepts,
            n_classes,
            explainer_hidden,
            conceptizator,
            temperature
    ):
        super(Explainer, self).__init__()
        self.model_layers = []
        self.model_layers.append(
            EntropyLinear(
                n_concepts,
                explainer_hidden[0],
                n_classes,
                temperature,
                conceptizator=conceptizator
            )
        )
        self.model_layers.append(torch.nn.LeakyReLU())
        for i in range(1, len(explainer_hidden)):
            self.model_layers.append(torch.nn.Linear(explainer_hidden[i - 1], explainer_hidden[i]))
            self.model_layers.append(torch.nn.LeakyReLU())
            # self.model_layers.append(Dropout())

        self.model_layers.append(torch.nn.Linear(explainer_hidden[-1], 1))
        self.model = torch.nn.Sequential(*self.model_layers)

    def forward(self, x):
        y_out = self.model(x).squeeze(-1)
        return y_out


