import argparse
import os
import sys

from Explainer.experiments_explainer_ham10k import test

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))

parser = argparse.ArgumentParser(description='SIIM-ISIC Training')
parser.add_argument('--data-root', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/SIIM-ISIC',
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
parser.add_argument(
    '--bb-dir', metavar='DIR',
    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/SIIM-ISIC/BB/lr_0.001_epochs_95_optim_SGD/Inception_V3',
    help='path to BB'
)
parser.add_argument('--seed', default=1, type=int, metavar='N', help='seed')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="SIIM-ISIC", help='dataset name')
parser.add_argument('--model-name', type=str, default="g_best_model_epoch_4", help='name of the checkpoint')
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

parser.add_argument('--concept_file_name', type=str, default='derma_ham10000_0.01_50.pkl', help="concept_file_name")
parser.add_argument('--iter', default=1, type=int, metavar='N', help='iteration')
parser.add_argument('--expert-to-train', default="explainer", type=str, metavar='N',
                    help='which expert to train? explainer or residual')
parser.add_argument('--cov', nargs='+', default=[0.3, 0.4], type=float, help='coverage of the dataset')
parser.add_argument('--alpha', default=0.5, type=float, help='trade off for Aux explainer using Selection Net')
parser.add_argument('--selection-threshold', default=0.5, type=float,
                    help='selection threshold of the selector for the test/val set')
parser.add_argument('--lr-residual', '--learning-rate-residual', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate of bb residual')
parser.add_argument('--momentum-residual', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight-decay-residual', type=float, default=1e-4, help='weight_decay for SGD')
parser.add_argument('--lr', '--learning-rate', nargs='+', default=[0.01, 0.001], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--input-size-pi', default=2048, type=int,
                    help='input size of pi - 2048 for layer4 (ResNet) or 1024 for layer3 (ResNet) ')
parser.add_argument('--temperature-lens', default=0.7, type=float, help='temperature for entropy layer in lens')
parser.add_argument('--lambda-lens', default=0.0001, type=float, help='weight for entropy loss')
parser.add_argument('--alpha-KD', default=0.9, type=float, help='weight for KD loss by Hinton')
parser.add_argument('--temperature-KD', default=10, type=float, help='temperature for KD loss')
parser.add_argument('--conceptizator', default='identity_bool', type=str, help='activation')
parser.add_argument('--hidden-nodes', nargs="+", default=[10], type=int, help='hidden nodes of the explainer model')
parser.add_argument('--epochs', type=int, default=500, help='batch size for training the explainer - g')
parser.add_argument('--epochs-residual', type=int, default=50, help='batch size for training the residual')
parser.add_argument('--concept-names', nargs='+',
                    default=[
                        # "Sex",
                        "BWV", "RegularDG", "IrregularDG", "RegressionStructures",
                        "IrregularStreaks", "RegularStreaks", "AtypicalPigmentNetwork", "TypicalPigmentNetwork"
                    ])
parser.add_argument('--lm', default=32.0, type=float, help='lagrange multiplier for selective KD loss')
parser.add_argument('--checkpoint-model', metavar='file', nargs="+",
                    default=['model_g_best_model_epoch_14.pth.tar'],
                    help='checkpoint file of GatedLogicNet')
parser.add_argument('--checkpoint-residual', metavar='file', nargs="+",
                    default=['model_residual_best_model_epoch_2.pth.tar'],
                    help='checkpoint file of residual')
parser.add_argument('--prev_explainer_chk_pt_folder', metavar='path', nargs="+",
                    default=[],
                    help='checkpoint file of residual')
parser.add_argument('--soft', default='y', type=str, metavar='N', help='soft/hard concept?')
parser.add_argument('--with_seed', default='n', type=str, metavar='N', help='trying diff seeds for paper')


def main():
    print("Test GLT for SIIM-ISIC")
    args = parser.parse_args()
    args.class_to_idx = {"benign": 0, "malignant": 1}
    test(args)


if __name__ == '__main__':
    main()
