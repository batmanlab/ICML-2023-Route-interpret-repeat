import os
import pickle
from collections import namedtuple
from itertools import product

import h5py
import numpy as np
import sklearn.metrics as metrics
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import lr_scheduler
from torchvision import transforms

from BB.models.BB_DenseNet121 import DenseNet121
from BB.models.BB_ResNet import ResNet
from BB.models.BB_ResNet50_metanorm import BB_ResNet50_metanorm
from BB.models.VIT import CONFIGS, VisionTransformer
from BB.models.VIT_p import VisionTransformer_projected
from scheduler import WarmupCosineSchedule, WarmupLinearSchedule


def create_lr_scheduler(optimizer, args):
    if args.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=args.epochs,
                                                   eta_min=args.min_lr)
    elif args.lr_scheduler == 'multistep':
        if args.steps == "":
            return None
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=args.steps,
                                             gamma=args.gamma)
    elif args.lr_scheduler == 'none':
        scheduler = None
    return scheduler


def compute_AUCs(gt, pred, n):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    AUROCs = []
    AUPRCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            AUPRCs.append(average_precision_score(gt_np[:, i], pred_np[:, i]))
        except:
            AUROCs.append(0.5)
            AUPRCs.append(0.5)
    return AUROCs, AUPRCs


def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs


def get_correct(y_hat, y, num_classes):
    if num_classes == 1:
        y_hat = [1 if y_hat[i] >= 0.5 else 0 for i in range(len(y_hat))]
        correct = [1 if y_hat[i] == y[i] else 0 for i in range(len(y_hat))]
        return np.sum(correct)
    else:
        return y_hat.argmax(dim=1).eq(y).sum().item()


def get_correct_multi_label(y_hat, y):
    y_hat = y_hat > 0.5
    correct = y.eq(y_hat).sum().item()
    return correct


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_runs(params):
    """
    Gets the run parameters using cartesian products of the different parameters.
    :param params: different parameters like batch size, learning rates
    :return: iterable run set
    """
    Run = namedtuple("Run", params.keys())
    runs = []
    for v in product(*params.values()):
        runs.append(Run(*v))

    return runs


def compute_AUROC(output_GT, output_pred, output_size):
    out_AUROC = []
    data_GT = output_GT.cpu().numpy()
    data_PRED = output_pred.cpu().numpy()

    for i in range(output_size):
        try:
            out_AUROC.append(metrics.roc_auc_score(data_GT[:, i], data_PRED[:, i]))
        except ValueError:
            pass
    return out_AUROC


def get_num_correct(y_hat, y):
    return y_hat.argmax(dim=1).eq(y).sum().item()


def cal_accuracy(label, out):
    return metrics.accuracy_score(label, out)


def cal_precision_multiclass(label, out):
    return metrics.precision_score(label, out, average='micro')


def cal_recall_multiclass(label, out):
    return metrics.recall_score(label, out, average='micro')


def cal_classification_report(label, out, labels):
    return metrics.classification_report(y_true=label, y_pred=out, target_names=labels, output_dict=True)


def dump_in_pickle(output_path, file_name, stats_to_dump):
    cls_report_file = open(os.path.join(output_path, file_name), "wb")
    pickle.dump(stats_to_dump, cls_report_file)
    cls_report_file.close()


def get_model(model_arch, dataset, pretrained, n_classes, layer=None):
    if model_arch == "ResNet101" or model_arch == "ResNet50":
        return ResNet(dataset=dataset, pre_trained=pretrained, n_class=n_classes, model_choice=model_arch, layer=layer)
    elif model_arch == "ViT-B_16":
        config = CONFIGS[model_arch]
        device = get_device()
        bb = VisionTransformer(
            config, 448, zero_head=True, num_classes=200, smoothing_value=0
        ).to(device)
        bb.load_state_dict(
            torch.load(
                os.path.join(args.checkpoints, args.dataset, "BB", args.root_bb, args.arch, args.checkpoint_bb),
            )["model"]
        )
        return bb


def get_model_explainer(args, device):
    if (args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152") and args.projected == "n":
        chkpt = os.path.join(args.checkpoints, args.dataset, 'BB', args.root_bb, args.arch, args.checkpoint_bb)
        print(f"==>> Loading BB from : {chkpt}")
        bb = ResNet(
            dataset=args.dataset, pre_trained=args.pretrained, n_class=len(args.labels),
            model_choice=args.arch, layer=args.layer
        ).to(device)
        bb.load_state_dict(
            torch.load(chkpt)
        )
        return bb
    elif args.arch == "ViT-B_16":
        config = CONFIGS[args.arch]
        chkpt = os.path.join(args.checkpoints, args.dataset, 'BB', args.root_bb, args.arch, args.checkpoint_bb)
        print(f"==>> Loading BB from : {chkpt}")
        bb = VisionTransformer(
            config, args.img_size, zero_head=True, num_classes=len(args.labels), smoothing_value=args.smoothing_value
        ).to(device)
        bb.load_state_dict(torch.load(chkpt)["model"])
        return bb
    elif args.arch == "ViT-B_16_projected":
        config = CONFIGS["ViT-B_16"]
        # chkpt = os.path.join(args.checkpoints, args.dataset, 'BB', args.root_bb, args.arch, args.checkpoint_bb)
        chkpt = os.path.join(
            "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/spurious-cub-specific-classes/cub/explainer/ViT-B_16/lr_0.01_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.95_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none/iter1/explainer_projected",
            "VIT_CUBS_8000_checkpoint.bin"
        )
        print(f"==>> Loading projected BB from : {chkpt}")
        bb = VisionTransformer_projected(
            config, args.img_size, zero_head=True, num_classes=len(args.labels), smoothing_value=args.smoothing_value
        ).to(device)
        bb.load_state_dict(torch.load(chkpt)["model"])
        return bb
    elif (args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "ResNet152") and args.projected == "y":
        dataset_path = os.path.join(args.output, args.dataset, args.dataset_folder_concepts)
        print(args.dataset_folder_concepts)
        print(dataset_path)
        attributes_train = torch.load(os.path.join(dataset_path, "train_attributes.pt")).numpy()
        target_train = torch.load(os.path.join(dataset_path, "train_class_labels.pt")).numpy()
        N = attributes_train.shape[0]
        X = np.zeros((N, 4))
        X[:, 0:4] = attributes_train[:, 108:112]
        XTX = np.transpose(X).dot(X)
        kernel = np.linalg.inv(XTX)
        cf_kernel = torch.nn.Parameter(torch.tensor(kernel).float().to(device), requires_grad=False)
        bb_projected = BB_ResNet50_metanorm(args, dataset_size=N, kernel=cf_kernel, train=False).to(device)
        chkpt = os.path.join(
            args.checkpoints, args.dataset, "explainer", args.arch, args.bb_projected, args.checkpoint_bb
        )
        bb_projected.load_state_dict(torch.load(chkpt))
        print(f"==>> Loading projected BB from : {chkpt}")
        return bb_projected
    elif args.arch == "densenet121":
        args.N_labels = len(args.labels)
        model_bb = DenseNet121(args, layer=args.layer)
        chk_pt_path_bb = os.path.join(
            args.checkpoints, args.dataset, "BB", args.root_bb, args.arch, args.disease_folder
        )
        model_chk_pt = torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_bb))
        model_bb.load_state_dict(model_chk_pt['state_dict'])
        return model_bb


def get_criterion(dataset):
    if dataset == "cub" or dataset == "awa2":
        return torch.nn.CrossEntropyLoss()


def get_optim(dataset, net, params):
    if dataset == "cub":
        return torch.optim.SGD(
            net.parameters(),
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=params["weight_decay"]
        )
    elif dataset == "awa2":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=params["lr"], weight_decay=params["weight_decay"]
        )


def get_scheduler(solver, args):
    if args.arch == "ResNet50" or args.arch == "ResNet101" or args.arch == "Inception_V3":
        return torch.optim.lr_scheduler.StepLR(solver, step_size=30, gamma=0.1)
    elif args.arch == "ViT-B_16":
        t_total = args.num_steps
        if args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(solver, warmup_steps=args.warmup_steps, t_total=t_total)
        else:
            scheduler = WarmupLinearSchedule(solver, warmup_steps=args.warmup_steps, t_total=t_total)

        return scheduler


def get_train_val_transforms(dataset, img_size, arch):
    if dataset == "cub" and (arch == "ResNet50" or arch == "ResNet101"):
        return {
            "train_transform": transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]),
            "val_transform": transforms.Compose([
                transforms.Resize(int(img_size / 0.875)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]),
            "save_transform": transforms.Compose([
                transforms.Resize(int(img_size / 0.875)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ])
        }
    elif dataset == "cub" and (arch == "ViT-B_16" or arch == "ViT-B_16_projected"):
        return {
            "train_transform": transforms.Compose([
                transforms.Resize((600, 600), Image.BILINEAR),
                transforms.RandomCrop((448, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]),
            "val_transform": transforms.Compose([
                transforms.Resize((600, 600), Image.BILINEAR),
                transforms.CenterCrop((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]),
            "save_transform": transforms.Compose([
                transforms.Resize(int(img_size / 0.875)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ])
        }
    elif dataset == "awa2" and (arch == "ResNet50" or arch == "ResNet101"):
        return {
            "train_transform": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    )
                    # transforms.ToPILImage(),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=(0.485, 0.456, 0.406),
                    #                      std=(0.229, 0.224, 0.225))
                ]),
            "val_transform": transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]),
            "save_transform": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        }

    elif dataset == "awa2" and (arch == "ViT-B_16" or arch == "ViT-B_16_projected"):
        return {
            "train_transform": transforms.Compose([
                transforms.Resize((600, 600), Image.BILINEAR),
                transforms.RandomCrop((448, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]),
            "val_transform": transforms.Compose([
                transforms.Resize((600, 600), Image.BILINEAR),
                transforms.CenterCrop((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]),
            "save_transform": transforms.Compose([
                transforms.Resize(int(img_size / 0.875)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ])
        }
    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        return transform


def get_test_transforms(dataset, img_size, arch):
    if dataset == "cub" and (arch == "ViT-B_16" or arch == "ViT-B_16_projected"):
        return transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    elif dataset == "cub" and (arch == "ResNet50" or arch == "ResNet101"):
        return transforms.Compose([
            transforms.Resize(int(img_size / 0.875)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])


def get_image_label(args, data_tuple, device):
    if args.dataset == "cub":
        if args.spurious_waterbird_landbird == "y":
            data, target, _ = data_tuple
        else:
            data, target, _ = data_tuple
        data, target = data.to(device), target.to(torch.long).to(device)
        return data, target
    elif args.dataset == "mnist":
        data, target = data_tuple
        data, target = data.to(device), target.to(torch.float32).to(device)
        target = target.reshape((target.shape[0], 1))
        return data, target


def get_image_attributes(data_tuple, spurious_waterbird_landbird, dataset_name):
    if dataset_name == "cub":
        data, _, attribute = data_tuple
        return data, attribute
    elif dataset_name == "mnist":
        data, attribute = data_tuple
        return data, attribute


def flatten_cnn_activations_using_max_pooled(activations, kernel_size, stride=1):
    max_pool = torch.nn.MaxPool2d(kernel_size, stride=1)
    torch_activation = torch.from_numpy(activations)
    max_pool_activation = max_pool(torch_activation)
    flatten_activations = max_pool_activation.view(
        max_pool_activation.size()[0], -1
    ).numpy()
    return flatten_activations


def flatten_cnn_activations_using_avg_pooled(activations, kernel_size, stride=1):
    avg_pool = torch.nn.AvgPool2d(kernel_size, stride=1)
    torch_activation = torch.from_numpy(activations)
    avg_pool_activation = avg_pool(torch_activation)
    flatten_activations = avg_pool_activation.view(
        avg_pool_activation.size()[0], -1
    ).numpy()
    return flatten_activations


def flatten_cnn_activations_using_activations(activations):
    flatten = torch.nn.Flatten()
    return flatten(activations)


def save_features(activations_path, activation_file, layer, activations):
    with h5py.File(os.path.join(activations_path, activation_file), 'w') as f:
        f.create_dataset(layer, data=activations)


def save_tensor(path, tensor_to_save):
    torch.save(tensor_to_save, path)


def save_np_array(path, arr_to_save):
    np.save(path, arr_to_save)


def replace_names(explanation: str, concept_names) -> str:
    """
    Replace names of concepts in a formula.

    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_abbr, f_name))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
