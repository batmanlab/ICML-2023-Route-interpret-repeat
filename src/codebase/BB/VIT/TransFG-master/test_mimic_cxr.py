# coding=utf-8
# sync test~1028 0439
from __future__ import absolute_import, division, print_function

import os
import pickle
import sys

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))
# from BB.models.VIT_densenet import VisionTransformer, CONFIGS
import argparse
import logging
import os
import random
import time
import warnings
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# from models.VIT import VisionTransformer, CONFIGS
from models.VIT_densenet import VisionTransformer, CONFIGS
from utils.data_utils import get_loader
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    pred_np = pred.detach().cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def save_model(args, model, chk_pt_path, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(chk_pt_path, f"{args.name}_{global_step}_checkpoint.bin")
    """
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    """
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", chk_pt_path)


def setup(args, chk_pt_path):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "cub":
        num_classes = 200
    if args.dataset == "mimic_cxr":
        num_classes = 2
    elif args.dataset == "DLCV_1":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    model = VisionTransformer(
        config, args.resize, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value,
        pretrained=args.pretrained
    )

    if args.checkpoint_file is not None:
        model_chk_pt = os.path.join(chk_pt_path, args.checkpoint_file)
        print(f"===> model checkpoint: {os.path.join(chk_pt_path, args.checkpoint_file)}")
        pretrained_model = torch.load(model_chk_pt)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, global_step, output_path, mode):
    # Validation!
    eval_losses = AverageMeter()
    feature_path = os.path.join(output_path, f"{mode}_features_VIT_full")
    os.makedirs(feature_path, exist_ok=True)
    print(feature_path)
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    val_out_put_pred = torch.FloatTensor().cuda()
    val_out_put_target = torch.FloatTensor().cuda()
    for step, batch in enumerate(epoch_iterator):
        (dicom_id,
         image,
         densenet_features,
         adj_mtx, _, _,
         landmark_spec_label,
         landmarks_spec_inverse_weight,
         landmark_spec_label_pnu,
         selected_obs_label_gt,
         selected_obs_inverse_weight,
         selected_obs_label_pnu, _, _, _, _, _) = batch
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)

        if torch.cuda.is_available():
            selected_obs_label_gt = selected_obs_label_gt.cuda(args.gpu, non_blocking=True)
            selected_obs_inverse_weight = selected_obs_inverse_weight.cuda(args.gpu, non_blocking=True)
            densenet_features = densenet_features.cuda(args.gpu, non_blocking=True).squeeze(dim=1)
            selected_obs_label_gt = selected_obs_label_gt.view(-1).to(torch.long)
            selected_obs_inverse_weight = selected_obs_inverse_weight.view(-1)

        with torch.no_grad():
            logits, part_tokens = model(densenet_features)
            eval_loss = loss_fct(logits, selected_obs_label_gt)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

        # print(part_tokens[:, 0].size())
        val_out_put_pred = torch.cat((val_out_put_pred, logits), dim=0)
        val_out_put_target = torch.cat((val_out_put_target, selected_obs_label_gt), dim=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
        torch.save(part_tokens.cpu(), os.path.join(feature_path, f"features_{step}.pth.tar"))

    proba = torch.nn.Softmax()(val_out_put_pred)[:, 1]
    auroc, aurpc = compute_AUC(gt=val_out_put_target, pred=proba)
    auroc = torch.tensor(auroc).to(args.device)
    # dist.barrier()
    val_auroc = auroc
    val_auroc = val_auroc.detach().cpu().numpy()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid auroc: %2.5f" % val_auroc)
    print(f"step: {global_step} || val_auroc: {val_auroc}")

    return val_auroc


def test(args, model):
    """ Test the model """
    args.shuffle = False
    args.train_batch_size = 1
    args.eval_batch_size = 1
    # Prepare dataset
    test_loader = get_loader(args, mode="test")
    train_loader, val_loader = get_loader(args, mode="train")

    global_step, best_auroc = 0, 0
    start_time = time.time()
    with torch.no_grad():
        feature_path = os.path.join(
            args.output,
            args.dataset,
            "t",
            args.dataset_folder_concepts,
            "densenet121",
            args.disease_folder,
            "dataset_g",
        )
        print("Validating testing dataset: ")
        auroc = valid(args, model, test_loader, global_step, feature_path, mode="test")
        print(f"Test Auroc: {auroc}")

        print("Validating training dataset: ")
        auroc = valid(args, model, train_loader, global_step, feature_path, mode="train")
        print(f"Train Auroc: {auroc}")

        print("Validating validating dataset: ")
        auroc = valid(args, model, val_loader, global_step, feature_path, mode="val")
        print(f"Val Auroc: {auroc}")

    logger.info("Best Auroc: \t%f" % best_auroc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f (hr)" % ((end_time - start_time) / 3600))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--logs', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to output logs')
    parser.add_argument("--name", default="VIT_mimic_cxr",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["mimic_cxr", "CUB_200_2011", "car", "dog", "nabirds", "INat2017", "DLCV_1"],
                        default="mimic_cxr",
                        help="Which dataset.")

    # parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/DL_CV/HW1/training_data/')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "ViT-B_32_densenet"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument(
        "--pretrained_dir", type=str,
        default="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/pretrained_VIT/ViT-B_16.npz",
        help="Where to search for pretrained ViT models."
    )
    parser.add_argument('--checkpoint-file', metavar='file',
                        default='VIT_CUBS_8000_checkpoint.bin',
                        help='checkpoint of the Blackbox')

    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    # parser.add_argument("--output_dir", default="./output", type=str,
    #                     help="The output directory where checkpoints will be written.")
    # Image Augmentation
    parser.add_argument('--resize', default=512, type=int,
                        help='input image resize')
    parser.add_argument('--crop', default=448, type=int,
                        help='resize image crop')
    parser.add_argument('--degree', default=10, type=int,
                        help='rotation range [-degree, +degree].')
    parser.add_argument('--mini-data', default=None, type=int, help='small dataset for debugging')
    # parser.add_argument("--img_size", default=448, type=int,
    #                     help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=500, type=int,
                        help="100Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    """
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    """

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    parser.add_argument('--exp-dir', metavar='DIR',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/exp/mimic_cxr/debug',
                        help='path to images')
    parser.add_argument('--loss', default='CE_W',
                        help='observation loss type.')

    # Dataset
    parser.add_argument('--image-path-ocean-shared', metavar='DIR',
                        default='/ocean/projects/asc170022p/shared/Projects/ImgTxtWeakSupLocalization/CXR_Datasets/MIMICCXR/files/mimic-cxr-jpg/2.0.0/files',
                        help='image path in ocean')

    parser.add_argument('--dataset-dir', metavar='DIR',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/',
                        help='dataset directory')
    parser.add_argument('--img-chexpert-file', metavar='PATH',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/mimic-cxr-chexpert.csv',
                        help='master table including the image path and chexpert labels.')
    parser.add_argument('--radgraph-adj-mtx-pickle-file', metavar='PATH',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/landmark_observation_adj_mtx_v2.pickle',
                        help='radgraph adjacent matrix landmark - observation.')
    parser.add_argument('--radgraph-sids-npy-file', metavar='PATH',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/landmark_observation_sids_v2.npy',
                        help='radgraph study ids.')
    parser.add_argument('--radgraph-adj-mtx-npy-file', metavar='PATH',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/landmark_observation_adj_mtx_v2.npy',
                        help='radgraph adjacent matrix landmark - observation.')
    parser.add_argument('--nvidia-bounding-box-file', metavar='PATH',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/mimic-cxr-annotation.csv',
                        help='bounding boxes annotated for pneumonia and pneumothorax.')
    parser.add_argument('--imagenome-bounding-box-file', metavar='PATH',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/',
                        help='ImaGenome bounding boxes for 21 landmarks.')
    parser.add_argument('--imagenome-radgraph-landmark-mapping-file', metavar='PATH',
                        default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/landmark_mapping.json',
                        help='Landmark mapping between ImaGenome and RadGraph.')
    parser.add_argument('--chexpert-names', nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                                                                'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                                                'Lung Opacity', 'No Finding', 'Pleural Effusion',
                                                                'Pleural Other', 'Pneumonia', 'Pneumothorax',
                                                                'Support Devices'])
    parser.add_argument('--full-anatomy-names', nargs='+',
                        default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                                 'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                                 'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe',
                                 'upper_left_lobe',
                                 'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung',
                                 'left_mid_lung', 'left_upper_lung',
                                 'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung',
                                 'right_upper_lung', 'right_apical_lung',
                                 'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic',
                                 'right_costophrenic', 'costophrenic_unspec',
                                 'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach',
                                 'right_atrium', 'right_ventricle', 'aorta', 'svc',
                                 'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary',
                                 'lung_volumes', 'unspecified', 'other'])
    parser.add_argument('--landmark-names-spec', nargs='+',
                        default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                                 'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                                 'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe',
                                 'upper_left_lobe',
                                 'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung',
                                 'left_mid_lung', 'left_upper_lung',
                                 'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung',
                                 'right_upper_lung', 'right_apical_lung',
                                 'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic',
                                 'right_costophrenic', 'costophrenic_unspec',
                                 'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium',
                                 'right_ventricle',
                                 'aorta', 'svc',
                                 'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
    parser.add_argument('--landmark-names-unspec', nargs='+',
                        default=['cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])
    parser.add_argument('--full-obs', nargs='+',
                        default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                                 'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                                 'process', 'abnormality', 'enlarge', 'tip', 'low',
                                 'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                                 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                                 'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire',
                                 'fluid',
                                 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                                 'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate',
                                 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                                 'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding',
                                 'borderline',
                                 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                                 'tail_abnorm_obs', 'excluded_obs'])
    parser.add_argument('--norm-obs', nargs='+',
                        default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                                 'expand', 'hyperinflate'])
    parser.add_argument('--abnorm-obs', nargs='+',
                        default=['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                                 'process', 'abnormality', 'enlarge', 'tip', 'low',
                                 'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                                 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                                 'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid',
                                 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                                 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass',
                                 'crowd',
                                 'infiltrate', 'obscure', 'deformity', 'hernia',
                                 'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding',
                                 'borderline',
                                 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
    parser.add_argument('--tail-abnorm-obs', nargs='+', default=['tail_abnorm_obs'])
    parser.add_argument('--excluded-obs', nargs='+', default=['excluded_obs'])
    parser.add_argument('--selected-obs', nargs='+', default=['pneumothorax'])
    parser.add_argument('--labels', nargs='+',
                        default=['0 (No Pneumothorax)', '1 (Pneumothorax)'])

    # PNU labels
    parser.add_argument('--landmark_label', default='PN',
                        help='anatomical landmark label type, PN or PUN.')
    parser.add_argument('--obs-u-alpha-method', default='discard',
                        help='how to deal with top alpha U samples? discard or replace (-1 to 1)')
    parser.add_argument('--warm-up', default=0, type=int,
                        help='number of epochs warm up.')
    parser.add_argument('--dataset-folder-concepts', type=str,
                        default="lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4",
                        help='dataset folder of concept bank')
    parser.add_argument('--pretrained', type=str,
                        default="n",
                        help='pretrained model for imagenet')

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    args.arch = args.model_type
    args.epochs = 60
    args.lr = args.learning_rate
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_labels = len(args.labels)
    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss}"
    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    args.disease_folder = disease_folder
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch, disease_folder)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch, disease_folder)

    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print(chk_pt_path)
    print(output_path)
    pickle.dump(args, open(os.path.join(output_path, "MIMIC_train_configs.pkl"), "wb"))
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(f"{output_path}", "VIT_Test.log"),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args, chk_pt_path)
    print("Initial Setup is done...")
    # Training
    test(args, model)


if __name__ == "__main__":
    main()
