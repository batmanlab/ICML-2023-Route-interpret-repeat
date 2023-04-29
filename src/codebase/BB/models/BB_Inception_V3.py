import os

import torch
import torch.nn as nn
import torchvision

MODEL_THRESHOLDS = {
    'HAM10000': 0.733,
    'g_best_model_epoch_4': 0.733,
}

MODEL_WEB_PATHS = {
    'HAM10000': 'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
    'g_best_model_epoch_4': 'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
}


def load_model(
        model_name, bb_dir="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/BB/Inception_V3",
        dataset="HAM10k"
):
    # Taken from the DDI repo https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP
    """Load the model and download if necessary. Saves model to provided save
    directory."""

    print(bb_dir)
    os.makedirs(bb_dir, exist_ok=True)
    model_path = None
    if dataset == "HAM10k":
        model_path = os.path.join(bb_dir, f"{model_name.lower()}.pth")
    elif dataset == "SIIM-ISIC":
        model_path = os.path.join(bb_dir, f"{model_name.lower()}.pth.tar")
    model = torchvision.models.inception_v3(init_weights=False, pretrained=False, transform_input=True)
    model.fc = torch.nn.Linear(2048, 2)
    model.AuxLogits.fc = torch.nn.Linear(768, 2)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model._ddi_name = model_name
    model._ddi_threshold = MODEL_THRESHOLDS[model_name]
    model._ddi_web_path = MODEL_WEB_PATHS[model_name]
    return model


class InceptionBottom(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionBottom, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        self.layer = layer
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.features = nn.Sequential(*all_children[:until_layer])
        self.model = original_model

    def _transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 model.x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 model.x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 model.x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 model.x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 model.x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192model. x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192model. x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256model. x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768model. x 17 x 17
        # N x 768model. x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 128model.0 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 204model.8 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 204model.8 x 8 x 8
        # Adaptivmodel.e average pooling
        x = self.model.avgpool(x)
        # N x 204model.8 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        return x


class InceptionTop(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionTop, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.layer = layer
        self.features = nn.Sequential(*all_children[until_layer:])

    def forward(self, x):
        logit = self.features(x)
        proba = nn.Softmax(dim=-1)(logit)
        return logit, proba


def get_model(bb_dir, model_name="ham10000"):
    model = load_model(model_name.upper(), bb_dir=bb_dir)
    model = model.to("cuda")
    model = model.eval()
    model_bottom, model_top = InceptionBottom(model), InceptionTop(model)
    return model, model_bottom, model_top


def get_model_isic(bb_dir, model_name="ham10000"):
    model = load_model(model_name.upper(), bb_dir=bb_dir)
    model = model.to("cuda")
    model_bottom, model_top = InceptionBottom(model), InceptionTop(model)
    return model, model_bottom, model_top


def get_BB_model_isic(bb_dir, model_name, dataset):
    model = load_model(model_name, bb_dir=bb_dir, dataset=dataset)
    model = model.to("cuda")
    model = model.eval()
    model_bottom, model_top = InceptionBottom(model), InceptionTop(model)
    return model, model_bottom, model_top


def main():
    print("Test BB for MIMIC_CXR")
    bb_dir = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/BB/Inception_V3"
    model, model_bottom, model_top = get_model(bb_dir, model_name="ham10000")
    print(model)


if __name__ == '__main__':
    main()
