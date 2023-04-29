import os
import pickle

import torch

import utils
from Explainer.models.explainer import Explainer

"""
    Inspired by Selective Net
"""


class Gated_Logic_Net(torch.nn.Module):

    def __init__(
            self,
            feature_dims,
            concept_names,
            labels,
            hidden_nodes,
            conceptizator,
            temperature_lens,
            init_weights=True,
            use_concepts_as_pi_input=True,
            as_baseline=False
    ):
        super(Gated_Logic_Net, self).__init__()
        self.use_concepts_as_pi_input = use_concepts_as_pi_input
        self.input_size = len(concept_names) if self.use_concepts_as_pi_input else feature_dims
        self.as_baseline = as_baseline
        # represented as f() in the original paper
        self.explainer = Explainer(
            n_concepts=len(concept_names),
            n_classes=len(labels),
            explainer_hidden=hidden_nodes,
            conceptizator=conceptizator,
            temperature=temperature_lens,
        )

        # represented as g() in the original paper
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.input_size),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.input_size),
            torch.nn.Linear(self.input_size, 1),
            torch.nn.Sigmoid(),
        )

        # represented as h() in the original paper
        self.aux_explainer = Explainer(
            n_concepts=len(concept_names),
            n_classes=len(labels),
            explainer_hidden=hidden_nodes,
            conceptizator=conceptizator,
            temperature=temperature_lens,
        )

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.selector)

    def forward(self, concept_x, feature_x=None, test=False):
        prediction_out = self.explainer(concept_x)
        if self.as_baseline:
            return prediction_out, self.explainer.model[0].concept_mask, \
                   self.explainer.model[0].alpha, self.explainer.model[0].alpha_norm, \
                   self.explainer.model[0].conceptizator

        auxiliary_out = self.aux_explainer(concept_x)
        if self.use_concepts_as_pi_input:
            selection_out = self.selector(concept_x)
        else:
            x_pi = self.adaptive_avg_pool(feature_x).reshape(-1, self.input_size * 1 * 1)
            selection_out = self.selector(x_pi)

        if test:
            return prediction_out, selection_out, auxiliary_out, self.explainer.model[0].concept_mask, \
                   self.explainer.model[0].alpha, self.explainer.model[0].alpha_norm, \
                   self.explainer.model[0].conceptizator
        else:
            return prediction_out, selection_out, auxiliary_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


def test():
    # use_concepts_as_pi_input = False
    # feature_x = torch.rand(10, 2048, 14, 14)
    # concept_x = torch.rand(10, 108)
    # print(concept_x.size())
    # model = Gated_Logic_Net(
    #     2048,
    #     torch.rand(108),
    #     torch.rand(200),
    #     [10],
    #     "identity_bool",
    #     10,
    #     use_concepts_as_pi_input=use_concepts_as_pi_input
    # )
    # prediction_out, selection_out, auxiliary_out = model(concept_x, feature_x)
    # print(prediction_out.size())
    # print(selection_out.size())
    # print(auxiliary_out.size())

    print("##########################")
    # use_concepts_as_pi_input = True
    # feature_x = torch.rand(10, 2048, 14, 14)
    concept_x = torch.rand(10, 108)
    # print(concept_x.size())
    # model = Gated_Logic_Net(
    #     2048,
    #     torch.rand(108),
    #     torch.rand(200),
    #     [10],
    #     "identity_bool",
    #     10,
    #     use_concepts_as_pi_input=use_concepts_as_pi_input
    # )
    #
    # # prediction_out, selection_out, auxiliary_out = model(concept_x)
    print("##########################")

    checkpoint_model = "model_g_best_model_epoch_159.pth.tar"
    pickle_in = open(
        os.path.join(
            "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/cub/explainer/lr_0.1_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none",
            "train_explainer_configs.pkl",
        ),
        "rb",
    )
    args = pickle.load(pickle_in)
    n_classes = len(args.labels)
    use_concepts_as_pi_input = (
        True
        if args.use_concepts_as_pi_input == "y" or args.use_concepts_as_pi_input
        else False
    )
    device = utils.get_device()
    print(f"Device: {device}")
    model = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
        use_concepts_as_pi_input,
    ).to(device)
    explainer_init = "none"
    root = (
        f"lr_{args.lr}_epochs_{args.epochs}_temperature-lens_{args.temperature_lens}"
        f"_use-concepts-as-pi-input_{use_concepts_as_pi_input}_input-size-pi_{args.input_size_pi}"
        f"_cov_{args.cov}_alpha_{args.alpha}_selection-threshold_{args.selection_threshold}"
        f"_lambda-lens_{args.lambda_lens}_alpha-KD_{args.alpha_KD}"
        f"_temperature-KD_{float(args.temperature_KD)}_hidden-layers_{len(args.hidden_nodes)}"
        f"_layer_{args.layer}_explainer_init_{explainer_init if not args.explainer_init else args.explainer_init}"
    )

    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "explainer", root)
    model.load_state_dict(torch.load(os.path.join(chk_pt_path, checkpoint_model)))
    model.eval()
    prediction_out, selection_out, auxiliary_out, concept_mask, \
    alpha, alpha_norm, conceptizator = model(concept_x.to(device), test=True)

    print(prediction_out.size())
    print(selection_out.size())
    print(auxiliary_out.size())
    print("model setting")
    print(concept_mask.size())
    print(alpha.size())
    print(alpha_norm.size())
    print(type(conceptizator))
    print(conceptizator.concepts.size())
    print(conceptizator.threshold)


if __name__ == "__main__":
    test()
