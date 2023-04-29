import argparse
import os
import sys

from BB.experiments_t_ham_10k import train_for_concepts

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))

parser = argparse.ArgumentParser(description='HAM10k Training')
parser.add_argument('--data-root', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/HAM10k',
                    help='path to dataset')
parser.add_argument('--logs', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log',
                    help='path to tensorboard logs')
parser.add_argument('--checkpoints', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                    help='path to checkpoints')
parser.add_argument('--output', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                    help='path to output logs')
parser.add_argument('--bb-dir', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/BB/Inception_V3',
                    help='path to BB')
parser.add_argument('--seed', default=42, type=int, metavar='N', help='seed')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="HAM10k", help='dataset name')
parser.add_argument('--model-name', type=str, default="ham10000", help='name of the checkpoint')
parser.add_argument(
    '--derm7_folder', type=str, default="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/Derm7pt",
    help='DERM7_Folder'
)
parser.add_argument('--derm7_meta', type=str, default="meta.csv", help='DERM7_META')
parser.add_argument('--derm7_train_idx', type=str, default="train_indexes.csv", help='TRAIN_IDX')
parser.add_argument('--derm7_val_idx', type=str, default="valid_indexes.csv", help='VAL_IDX')
parser.add_argument('--img-size', type=int, default=448, help='image\'s size for transforms')
parser.add_argument(
    '--bs', '--batch-size', '--train_batch_size', '--eval_batch_size', default=32, type=int,
    metavar='N', help='batch size BB'
)

parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--eval_every", default=100, type=int,
                    help="100Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight_decay for SGD')
parser.add_argument('--epochs', type=int, default=95, help='batch size for training')
parser.add_argument('--arch', type=str, default="Inception_V3", help='Architecture of BB')
parser.add_argument("--name", default="VIT_CUBS",
                    help="Name of this run. Used for monitoring.")
parser.add_argument(
    "--pretrained_dir", type=str,
    default="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/pretrained_VIT/ViT-B_16.npz",
    help="Where to search for pretrained ViT models."
)
parser.add_argument("--pretrained_model", type=str, default=None,
                    help="load pretrained model")
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
parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--smoothing_value', type=float, default=0.0,
                    help="Label smoothing value\n")

parser.add_argument('--split', type=str, default='non-overlap',
                    help="Split method")
parser.add_argument('--slide_step', type=int, default=12,
                    help="Slide step for overlap split")
parser.add_argument('--labels', nargs='+',
                    default=['0 (Benign)', '1 (Malignant)'])

parser.add_argument("--n-samples", default=50, type=int)
parser.add_argument("--C", nargs="+", type=float, default=[0.01])
parser.add_argument('--concepts', nargs='+',
                    default=[
                        # "Sex",
                        "BWV", "RegularDG", "IrregularDG", "RegressionStructures",
                        "IrregularStreaks", "RegularStreaks", "AtypicalPigmentNetwork", "TypicalPigmentNetwork"
                    ])


def main():
    print("Test T for HAM10k")
    args = parser.parse_args()
    args.class_to_idx = {"benign": 0, "malignant": 1}
    print(args.concepts)
    train_for_concepts(args)


if __name__ == '__main__':
    main()
