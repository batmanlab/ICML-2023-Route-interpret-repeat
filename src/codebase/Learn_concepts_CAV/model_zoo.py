import os

import numpy as np
import torch

import utils
from BB.models.BB_ResNet import ResNet
from BB.models.BB_ResNet50_metanorm import BB_ResNet50_metanorm
from BB.models.VIT import VisionTransformer, CONFIGS
from BB.models.VIT_p import VisionTransformer_projected


def get_model(args):
    device = utils.get_device()
    chk_pt_path_bb = os.path.join(
        args.checkpoints, args.dataset, "BB", f"lr_{args.lr}_epochs_{args.epochs}", args.arch
    )
    print(f"Device: {device}")
    if (
            args.arch == "ResNet50" or
            args.arch == "ResNet101" or
            args.arch == "ResNet152"
    ) and args.projected == "n":
        bb = ResNet(
            dataset=args.dataset,
            pre_trained=args.pretrained,
            n_class=len(args.labels),
            model_choice=args.arch,
            layer=args.layer
        ).to(device)
        print(f"BB is loaded from {os.path.join(chk_pt_path_bb, args.checkpoint_file)}")
        bb.load_state_dict(torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_file)))
        return bb

    elif (
            args.arch == "ResNet50" or
            args.arch == "ResNet101" or
            args.arch == "ResNet152"
    ) and args.projected == "y":
        dataset_path = os.path.join(
            args.output, args.dataset, "t", args.dataset_folder_concepts, "dataset_g"
        )
        attributes_train = torch.load(os.path.join(dataset_path, "train_attributes.pt")).numpy()
        N = attributes_train.shape[0]
        X = np.zeros((N, 4))
        X[:, 0:4] = attributes_train[:, 108:112]
        XTX = np.transpose(X).dot(X)
        kernel = np.linalg.inv(XTX)
        cf_kernel = torch.nn.Parameter(torch.tensor(kernel).float().to(device), requires_grad=False)

        bb = BB_ResNet50_metanorm(args, dataset_size=N, kernel=cf_kernel, train=False).to(device)
        chkpt = os.path.join(chk_pt_path_bb, args.checkpoint_file)

        print(f"Projected CNN BB is loaded from {chkpt}")
        bb.load_state_dict(torch.load(chkpt))
        return bb

    elif args.arch == "ViT-B_16":
        config = CONFIGS[args.arch]
        print(f"BB is loaded from {os.path.join(chk_pt_path_bb, args.checkpoint_file)}")
        bb = VisionTransformer(
            config, args.img_size, zero_head=True, num_classes=len(args.labels),
            smoothing_value=args.smoothing_value
        ).to(device)
        bb.load_state_dict(torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_file))["model"])
        return bb

    elif args.arch == "ViT-B_16_projected":
        config = CONFIGS["ViT-B_16"]
        # chkpt = os.path.join(args.checkpoints, args.dataset, 'BB', args.root_bb, args.arch, args.checkpoint_bb)
        chkpt = os.path.join(
            "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/spurious-cub-specific-classes/cub/explainer/ViT-B_16/lr_0.01_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.95_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none/iter1/explainer_projected",
            "VIT_CUBS_8000_checkpoint.bin"
        )
        print(f"==>> Loading projected VIT BB from : {chkpt}")
        bb = VisionTransformer_projected(
            config, args.img_size, zero_head=True, num_classes=len(args.labels), smoothing_value=args.smoothing_value,
            get_phi=True
        ).to(device)
        bb.load_state_dict(torch.load(chkpt)["model"])
        return bb
