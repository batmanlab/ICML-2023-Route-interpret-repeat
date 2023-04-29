# coding=utf-8
# sync test~1028 0439
from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))

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

from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader

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
        config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value
    )

    checkpoint = np.load(args.pretrained_dir)
    model.load_from(checkpoint)
    if args.checkpoint_file is not None:
        model_chk_pt = os.path.join(chk_pt_path, args.checkpoint_file)
        print(model_chk_pt)
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


def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

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
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y, _ = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    # dist.barrier()
    val_accuracy = accuracy
    val_accuracy = val_accuracy.detach().cpu().numpy()
    np.save(os.path.join(args.output, "all_preds.npy"), all_preds)
    np.save(os.path.join(args.output, "all_label.npy"), all_label)
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Auroc: %2.5f" % val_accuracy)
    print(f"step: {global_step} || val_acc: {val_accuracy * 100}")
    return val_accuracy


def test(args, model):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir, exist_ok=True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    test_loader = get_loader(args, mode="test", waterbird_landbird=True)
    global_step, best_acc = 0, 0
    start_time = time.time()
    with torch.no_grad():
        accuracy = valid(args, model, test_loader, global_step)

    print(f"Test_acc: {accuracy * 100}")
    logger.info("Best Auroc: \t%f" % accuracy)
    logger.info("End Testing!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--data-root', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/spurious/waterbird_complete95_forest2water2',
                        help='path to dataset')
    parser.add_argument('--json-root', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/data_preprocessing',
                        help='path to json files containing train-val-test split')
    parser.add_argument('--logs', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log/spurious-cub-waterbird-landbird',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/spurious-cub-waterbird-landbird',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/spurious-cub-waterbird-landbird',
                        help='path to output logs')
    parser.add_argument('--attribute-file-name', metavar='file',
                        default='attributes_spurious.npy',
                        help='file containing all the concept attributes')
    parser.add_argument("--name", default="VIT_CUBS",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "DLCV_1"],
                        default="cub",
                        help="Which dataset.")

    # parser.add_argument('--data_root', type=str, default='/home/skchen/ML_practice/DL_CV/HW1/training_data/')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument(
        "--pretrained_dir", type=str,
        default="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/pretrained_VIT/ViT-B_16.npz",
        help="Where to search for pretrained ViT models."
    )
    parser.add_argument('--checkpoint-file', metavar='file',
                        default='VIT_CUBS_3000_checkpoint.bin',
                        help='checkpoint file of BB')

    # parser.add_argument("--pretrained_model", type=str, default=None,
    #                     help="load pretrained model")
    # parser.add_argument("--output_dir", default="./output", type=str,
    #                     help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=200, type=int,
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
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

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
    parser.add_argument('--concept-names', nargs='+',
                        default=['has_bill_shape_dagger', 'has_bill_shape_hooked_seabird',
                                 'has_bill_shape_allpurpose', 'has_bill_shape_cone', 'has_wing_color_brown',
                                 'has_wing_color_grey', 'has_wing_color_yellow', 'has_wing_color_black',
                                 'has_wing_color_white', 'has_wing_color_buff', 'has_upperparts_color_brown',
                                 'has_upperparts_color_grey', 'has_upperparts_color_yellow',
                                 'has_upperparts_color_black', 'has_upperparts_color_white',
                                 'has_upperparts_color_buff', 'has_underparts_color_brown',
                                 'has_underparts_color_grey', 'has_underparts_color_yellow',
                                 'has_underparts_color_black', 'has_underparts_color_white',
                                 'has_underparts_color_buff', 'has_breast_pattern_solid',
                                 'has_breast_pattern_striped', 'has_breast_pattern_multicolored',
                                 'has_back_color_brown', 'has_back_color_grey', 'has_back_color_yellow',
                                 'has_back_color_black', 'has_back_color_white', 'has_back_color_buff',
                                 'has_tail_shape_notched_tail', 'has_upper_tail_color_brown',
                                 'has_upper_tail_color_grey', 'has_upper_tail_color_black',
                                 'has_upper_tail_color_white', 'has_upper_tail_color_buff',
                                 'has_head_pattern_plain', 'has_head_pattern_capped',
                                 'has_breast_color_brown', 'has_breast_color_grey',
                                 'has_breast_color_yellow', 'has_breast_color_black',
                                 'has_breast_color_white', 'has_breast_color_buff', 'has_throat_color_grey',
                                 'has_throat_color_yellow', 'has_throat_color_black',
                                 'has_throat_color_white', 'has_eye_color_black',
                                 'has_bill_length_about_the_same_as_head',
                                 'has_bill_length_shorter_than_head', 'has_forehead_color_blue',
                                 'has_forehead_color_brown', 'has_forehead_color_grey',
                                 'has_forehead_color_yellow', 'has_forehead_color_black',
                                 'has_forehead_color_white', 'has_forehead_color_red',
                                 'has_under_tail_color_brown', 'has_under_tail_color_grey',
                                 'has_under_tail_color_yellow', 'has_under_tail_color_black',
                                 'has_under_tail_color_white', 'has_under_tail_color_buff',
                                 'has_nape_color_blue', 'has_nape_color_brown', 'has_nape_color_grey',
                                 'has_nape_color_yellow', 'has_nape_color_black', 'has_nape_color_white',
                                 'has_nape_color_buff', 'has_belly_color_grey', 'has_belly_color_yellow',
                                 'has_belly_color_black', 'has_belly_color_white', 'has_belly_color_buff',
                                 'has_wing_shape_roundedwings', 'has_size_small_5__9_in',
                                 'has_size_medium_9__16_in', 'has_size_very_small_3__5_in',
                                 'has_shape_perchinglike', 'has_back_pattern_solid',
                                 'has_back_pattern_striped', 'has_back_pattern_multicolored',
                                 'has_tail_pattern_solid', 'has_tail_pattern_multicolored',
                                 'has_belly_pattern_solid', 'has_primary_color_brown',
                                 'has_primary_color_grey', 'has_primary_color_yellow',
                                 'has_primary_color_black', 'has_primary_color_white',
                                 'has_primary_color_buff', 'has_leg_color_grey', 'has_leg_color_black',
                                 'has_leg_color_buff', 'has_bill_color_grey', 'has_bill_color_black',
                                 'has_crown_color_blue', 'has_crown_color_brown', 'has_crown_color_grey',
                                 'has_crown_color_yellow', 'has_crown_color_black', 'has_crown_color_white',
                                 'has_wing_pattern_solid', 'has_wing_pattern_striped',
                                 'has_wing_pattern_multicolored',
                                 'has_ocean', 'has_lake', 'has_bamboo', 'has_forest'])

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
    args.epochs = 95
    args.lr = args.learning_rate
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    args.checkpoints = f"{args.checkpoints}-{args.img_size}"
    args.output = f"{args.output}-{args.img_size}"
    args.logs = f"{args.logs}-{args.img_size}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch)
    args.labels = ["0 (Landbird)", "1 (Waterbird)"]
    args.output = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/spurious-cub-waterbird-landbird-{args.img_size}/cub/BB/lr_0.03_epochs_95/ViT-B_16"
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(f"{output_path}", "VIT_test.log"),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seed
    set_seed(args)
    print(os.path.join(f"{output_path}", "VIT_test.log"))
    # Model & Tokenizer Setup
    args, model = setup(args, chk_pt_path)
    # Training
    test(args, model)


if __name__ == "__main__":
    main()
