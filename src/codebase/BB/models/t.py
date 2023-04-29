import torch


class Logistic_Regression_t(torch.nn.Module):
    def __init__(self, ip_size, op_size, flattening_type="adaptive"):
        super(Logistic_Regression_t, self).__init__()
        self.ip_size = ip_size
        self.flattening_type = flattening_type
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=ip_size, out_features=op_size, bias=True)

    def forward(self, x):
        if self.flattening_type == "adaptive":
            x_avg_pool = self.adaptive_avg_pool(x).reshape(-1, self.ip_size * 1 * 1)
            logits = self.linear(x_avg_pool)
            return logits
        elif self.flattening_type == "projected":
            logits = self.linear(x)
            return logits
        elif self.flattening_type == "flatten":
            logits = self.linear(self.flatten(x))
            return logits
        elif self.flattening_type == "max_pool":
            x_m_pool = self.adaptive_max_pool(x).reshape(-1, self.ip_size * 1 * 1)
            logits = self.linear(x_m_pool)
            return logits
        elif self.flattening_type == "VIT":
            logits = self.linear(x)
            return logits


if __name__ == "__main__":
    # x = torch.rand(16, 1024, 16, 16)
    # model = Logistic_Regression_t(ip_size=1024 * 16 * 16, op_size=107, flattening_type="flatten")
    # # print(model(x))
    # print(model(x).size())
    #
    # x = torch.rand(16, 512, 16, 16)
    # model = Logistic_Regression_t(ip_size=512 * 16 * 16, op_size=46, flattening_type="flatten")
    # # print(model(x))
    # print(model(x).size())

    x = torch.rand(16, 17408)
    model = Logistic_Regression_t(ip_size=17408, op_size=107, flattening_type="vit_flatten")
    # print(model(x))
    print(model(x).size())
    #
    # print(model.parameters())
    # for name, params in model.named_parameters():
    #     print(name, params)
    #     print(params.size())
