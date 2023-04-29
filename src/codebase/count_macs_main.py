import argparse
import copy
import json
import os
import pickle
import sys

import torch
import torch.fx
from thop import clever_format
from thop import profile

from BB.models.BB_DenseNet121 import DenseNet121
from BB.models.BB_Inception_V3 import get_model
from BB.models.BB_ResNet import ResNet
from BB.models.VIT import CONFIGS, VisionTransformer
from Explainer.models.residual import Residual

torch.fx.wrap('len')
# torch.fx.wrap('sqrt')
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


#################################
# python count_macs_main.py --dataset cub --arch ResNet101
# macs, params
# Metrics for MoIE
# macs: 16.204K, params: 12.119K
# iter: 1, coverage: 0.20844831246019954
# iter: 2, coverage: 0.12831670558267882
# iter: 3, coverage: 0.13712587561027384
# iter: 4, coverage: 0.13712587561027384
# iter: 5, coverage: 0.12088728507747824
# iter: 6, coverage: 0.14275100827849713

# Metrics for Residuals
# [INFO] Register count_adap_avgpool() for <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>.
# [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
# Metrics for Residual
# macs: 813.056K, params: 409.800K

# python count_macs_main.py --dataset cub --arch ViT-B_16
# Metrics for MoIE
# macs: 16.204K, params: 12.119K
# iter: 1, coverage: 0.16886011462534495
# iter: 2, coverage: 0.17926130333262577
# iter: 3, coverage: 0.15485035024410954
# iter: 4, coverage: 0.1440246232222458
# iter: 5, coverage: 0.1346847803014222
# iter: 6, coverage: 0.17639566970919127
# Metrics for Residuals
# [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
# Metrics for Residual
# macs: 153.600K, params: 153.800K

# python count_macs_main.py --dataset HAM10k --arch Inception_V3
# macs: 144.000B, params: 119.000B
# /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/BB/Inception_V3
# [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
# [INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
# Metrics for Residual
# macs: 4.096K, params: 4.098K

# python count_macs_main.py --dataset awa2 --arch ResNet50 --iterations 4
# Metrics for MoIE
# macs: 8.650K, params: 7.588K
# iter: 1, coverage: 0.3204273704658874
# iter: 2, coverage: 0.23887195632514988
# iter: 3, coverage: 0.19114445523662793
# iter: 4, coverage: 0.08550758616069934
# Metrics for Residuals
# [INFO] Register count_adap_avgpool() for <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>.
# [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
# Metrics for Residual
# macs: 801.056K, params: 398.800K

# python count_macs_main.py --dataset awa2 --arch ViT-B_16
# Metrics for MoIE
# macs: 8.650K, params: 7.588K
# iter: 1, coverage: 0.19221623069966842
# iter: 2, coverage: 0.21931205412466087
# iter: 3, coverage: 0.19114445523662793
# iter: 4, coverage: 0.13403891884650165
# iter: 5, coverage: 0.12178048698797601
# iter: 6, coverage: 0.08919181431490103
# Metrics for Residuals
# [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
# Metrics for Residual
# macs: 38.400K, params: 38.450K

# iter: 1, coverage: 0.3886669995007489
# iter: 2, coverage: 0.12343984023964054
# iter: 3, coverage: 0.14590614078881678
# iter: 4, coverage: 0.10172241637543684
# iter: 5, coverage: 0.0717673489765352
# iter: 6, coverage: 0.06927109335996005


##################################

def get_bb(args):
    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152":
        bb = ResNet(
            dataset=args.dataset, pre_trained=args.pretrained, n_class=len(args.labels),
            model_choice=args.arch, layer=args.layer
        ).to(device)
        return bb
    elif args.arch == "ViT-B_16":
        config = CONFIGS[args.arch]
        bb = VisionTransformer(
            config, args.img_size, zero_head=True, num_classes=len(args.labels), smoothing_value=args.smoothing_value
        ).to(device)
        return bb
    elif args.arch == "densenet121":
        args.N_labels = len(args.labels)
        model_bb = DenseNet121(args, layer=args.layer)
        return model_bb


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_path', metavar='DIR', default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022',
        help='path to checkpoints'
    )
    parser.add_argument(
        '--output', metavar='DIR', default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
        help='path to output logs'
    )
    parser.add_argument('--arch', type=str, default="ViT-B_16", help='Architecture of the blackbox')
    parser.add_argument('--dataset', type=str, default="cub", help='dataset name')
    parser.add_argument('--iterations', default=6, type=int, help='iterations for MoIE')
    parser.add_argument('--cur_iter', default=1, type=int, help='iterations for MoIE')
    parser.add_argument('--top_K', nargs='+', default=[3, 5, 10, 15, 20, 25, 30], type=int, help='How many concepts?')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE or Baseline_CBM_logic or Baseline_PCBM_logic')
    parser.add_argument('--expert', default='expert', type=str, help='expert or residual')

    return parser.parse_args()


def cal_performance_moIE(args, pkl, device):
    moIE = Gated_Logic_Net(
        pkl.input_size_pi,
        pkl.concept_names,
        pkl.labels,
        pkl.hidden_nodes,
        pkl.conceptizator,
        pkl.temperature_lens,
    ).to(device)

    inp = torch.rand(1, len(pkl.concept_names)).to(device)
    macs, params = profile(moIE, inputs=(inp,))
    # print(macs, params)
    macs, params = clever_format([macs, params], "%.3f")
    print("Metrics for MoIE")
    print(f"macs: {macs}, params: {params}")
    tot_size = 0
    if args.dataset == "cub":
        tot_size = 9422
    elif args.dataset == "awa2":
        tot_size = 29857
    elif args.dataset == "HAM10k":
        tot_size = 8012


    for _iter in range(args.iterations):
        _iter += 1
        output = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{_iter}"]["output"]
        prev_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{_iter}"]["prev_path"]
        explainer = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{_iter}"]["explainer_path"]
        full_output_path = os.path.join(
            args.output, args.dataset, "explainer", base_path, prev_path, f"iter{_iter}",
            explainer, output
        )
        files = paths[f"{args.dataset}_{args.arch}"]["files"]
        tensor_y = torch.load(os.path.join(full_output_path, files["train_tensor_y"]))
        print(f"iter: {_iter}, coverage: {tensor_y.size(0) / tot_size}")


def cal_performance_residual(args, pkl, device):
    feature_x = None
    input = torch.randn(1, 3, 448, 448).to(device)
    if args.dataset == "cub" or args.dataset == "awa2":
        bb = get_bb(pkl)
        residual = Residual(args.dataset, pkl.pretrained, len(pkl.labels), args.arch).to(device)
        if pkl.arch == "ResNet50" or pkl.arch == "ResNet101" or pkl.arch == "ResNet152":
            _ = bb(input)
            # feature_x = get_flattened_x(bb.feature_store[layer], flattening_type)
            feature_x = bb.feature_store[pkl.layer]
        elif pkl.arch == "ViT-B_16":
            _, tokens = bb(input)
            feature_x = tokens[:, 0]
        elif pkl.arch == "Inception_V3":
            feature_x = bb(input)


    elif args.dataset == "HAM10k":
        _, bb_model_bottom, bb_model_top = get_model(pkl.bb_dir, pkl.model_name)
        feature_x = bb_model_bottom(input)
        residual = copy.deepcopy(bb_model_top)

    # print(feature_x.size())
    # print(residual)
    macs, params = profile(residual, inputs=(feature_x,))
    # print(macs, params)
    macs, params = clever_format([macs, params], "%.3f")
    print("Metrics for Residual")
    print(f"macs: {macs}, params: {params}")

    tot_size = 0
    if args.dataset == "cub":
        tot_size = 9422
    elif args.dataset == "awa2":
        tot_size = 29857
    elif args.dataset == "HAM10k":
        tot_size = 8012

    # for _iter in range(args.iterations):
    #     _iter += 1
    #     prev_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{_iter}"]["prev_path"]
    #     full_output_path = os.path.join(
    #         args.output, args.dataset, "explainer", base_path, prev_path, f"iter{_iter}", "bb/residual_outputs"
    #     )
    #     files = paths[f"{args.dataset}_{args.arch}"]["files"]
    #     tensor_y = torch.load(os.path.join(full_output_path, files["train_tensor_y"]))
    #     print(f"iter: {_iter}, coverage: {tensor_y.size(0) / tot_size}")


if __name__ == "__main__":
    args = config()
    args.json_file = os.path.join(args.base_path, "codebase", "Completeness_and_interventions", "paths_MoIE.json")
    # model = resnet50()
    # input = torch.randn(1, 3, 224, 224)
    # macs, params = profile(model, inputs=(input,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs)

    device = 'cuda:0'
    with open(args.json_file) as _file:
        paths = json.load(_file)
    base_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["base_path"]
    prev_path = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["prev_path"]
    explainer = paths[f"{args.dataset}_{args.arch}"]["MoIE_paths"][f"iter{args.cur_iter}"]["explainer_path"]
    test_config = os.path.join(
        args.output, args.dataset, "explainer", base_path, prev_path, f"iter{args.cur_iter}",
        explainer, "test_explainer_configs.pkl",
    )
    pkl = pickle.load(open(test_config, "rb"))
    cal_performance_moIE(args, pkl, device)
    cal_performance_residual(args, pkl, device)
