import argparse
import os
import sys

from Explainer.completeness_score import cal_completeness_score, cal_completeness_stats, cal_completeness_stats_per_iter
from Explainer.experiments_explainer_awa2 import train_glt

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))

parser = argparse.ArgumentParser(description='Concept completeness Training')
parser.add_argument('--data-root', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/CUB_200_2011',
                    help='path to dataset')
parser.add_argument('--json-root', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/data_preprocessing',
                    help='path to json files containing train-val-test split')
parser.add_argument('--logs', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log',
                    help='path to tensorboard logs')
parser.add_argument('--checkpoints', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                    help='path to checkpoints')
parser.add_argument('--checkpoint-model', metavar='file', nargs="+",
                    default=['model_g_best_model_epoch_116.pth.tar'],
                    help='checkpoint file of the model GatedLogicNet')
parser.add_argument('--checkpoint-t-path', metavar='file',
                    default="lr_0.03_epochs_95/lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE",
                    help='checkpoint file of residual')
parser.add_argument('--root-bb', metavar='file',
                    default='lr_0.001_epochs_95',
                    help='checkpoint folder of BB')
parser.add_argument('--checkpoint-bb', metavar='file',
                    default='best_model_epoch_63.pth.tar',
                    help='checkpoint file of BB')
parser.add_argument('--checkpoint-file-t', metavar='file',
                    default='g_best_model_epoch_200.pth.tar',
                    help='checkpoint file of t')
parser.add_argument('--output', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                    help='path to output logs')
parser.add_argument('--attribute-file-name', metavar='file',
                    default='attributes.npy',
                    help='file containing all the concept attributes')
parser.add_argument('--iter', default=2, type=int, metavar='N', help='iteration')
parser.add_argument('--expert-to-train', default="explainer", type=str, metavar='N',
                    help='which expert to train? explainer or residual')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="cub", help='dataset name')
parser.add_argument('--img-size', type=int, default=448, help='image\'s size for transforms')
parser.add_argument('--cov', nargs='+', default=[0.45, 0.4], type=float, help='coverage of the dataset')
parser.add_argument('--alpha', default=0.5, type=float, help='trade off for Aux explainer using Selection Net')
parser.add_argument('--selection-threshold', default=0.5, type=float,
                    help='selection threshold of the selector for the test/val set')
parser.add_argument('--use-concepts-as-pi-input', default="y", type=str,
                    help='Input for the pi - Concepts or features? y for concepts else features')
parser.add_argument('--bs', '--batch-size', default=4, type=int, metavar='N', help='batch size BB')
parser.add_argument('--dataset-folder-concepts', type=str,
                    default="lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE",
                    help='dataset folder of concepts')
parser.add_argument('--lr-residual', '--learning-rate-residual', default=0.001, type=float,
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
parser.add_argument('--hidden-nodes',  default=10, type=int, help='hidden nodes of the explainer model')
parser.add_argument('--explainer-init', default=None, type=str, help='Initialization of explainer')
parser.add_argument('--epochs', type=int, default=100, help='batch size for training the explainer - g')
parser.add_argument('--epochs-residual', type=int, default=50, help='batch size for training the residual')
parser.add_argument('--layer', type=str, default="layer4", help='batch size for training of t')
parser.add_argument('--arch', type=str, default="ResNet101", required=True, help='ResNet50 or ResNet101 or ResNet152')
parser.add_argument('--smoothing_value', type=float, default=0.0,
                    help="Label smoothing value\n")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=10000, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument('--flattening-type', type=str, default="adaptive", help='flatten or adaptive or maxpool')
parser.add_argument('--solver-LR', type=str, default="sgd", help='solver - sgd/adam')
parser.add_argument('--loss-LR', type=str, default="BCE", help='loss - focal/BCE')
parser.add_argument('--prev_explainer_chk_pt_folder', metavar='path', nargs="+",
                    default=[
                        "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ViT-B_16/lr_0.01_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none/iter1",
                    ],
                    help='checkpoint file of residual')

parser.add_argument('--train_baseline', type=str, default="n", help='train baseline or glt')

parser.add_argument('--concept-names', nargs='+',
                    default=["black",
                             "white", "blue", "brown", "gray", "orange", "red", "yellow", "patches", "spots", "stripes",
                             "furry", "hairless", "toughskin", "big", "small", "bulbous", "lean", "flippers", "hands",
                             "hooves", "pads", "paws", "longleg", "longneck", "tail", "chewteeth", "meatteeth",
                             "buckteeth", "strainteeth", "horns", "claws", "tusks", "smelly", "flys", "hops", "swims",
                             "tunnels", "walks", "fast", "slow", "strong", "weak", "muscle", "bipedal", "quadrapedal",
                             "active", "inactive", "nocturnal", "hibernate", "agility", "fish", "meat", "plankton",
                             "vegetation", "insects", "forager", "grazer", "hunter", "scavenger", "skimmer", "stalker",
                             "newworld", "oldworld", "arctic", "coastal", "desert", "bush", "plains", "forest",
                             "fields", "jungle", "mountains", "ocean", "ground", "water", "tree", "cave", "fierce",
                             "timid", "smart", "group", "solitary", "nestspot", "domestic",
                             ])
parser.add_argument('--labels', nargs='+',
                    default=[
                        "antelope", "grizzly+bear", "killer+whale", "beaver", "dalmatian", "persian+cat", "horse",
                        "german+shepherd", "blue+whale", "siamese+cat", "skunk", "mole", "tiger", "hippopotamus",
                        "leopard", "moose", "spider+monkey", "humpback+whale", "elephant", "gorilla", "ox",
                        "fox", "sheep", "seal", "chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat",
                        "giraffe", "wolf", "chihuahua", "rat", "weasel", "otter", "buffalo", "zebra", "giant+panda",
                        "deer", "bobcat", "pig", "lion", "mouse", "polar+bear", "collie",
                        "walrus", "raccoon", "cow", "dolphin"
                    ])

parser.add_argument('--spurious-specific-classes', type=str, default="n", required=False, help='y or n')
parser.add_argument('--spurious-waterbird-landbird', type=str, default="n", required=False, help='y or n')
parser.add_argument('--bb-projected', metavar='file',
                    default='_cov_1.0/iter1/bb_projected/batch_norm_n_finetune_y',
                    help='checkpoint folder of BB')
parser.add_argument('--projected', type=str, default="n", required=False, help='n')
parser.add_argument('--soft', default='y', type=str, metavar='N', help='soft/hard concept?')
parser.add_argument('--with_seed', default='n', type=str, metavar='N', help='trying diff seeds for paper')

parser.add_argument('--g_lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate of g for completeness')

parser.add_argument('--per_iter_completeness', default='n', type=str, metavar='N',
                    help='Compute completeness per iteration or as a whole')
parser.add_argument('--g_checkpoint', default='g_best_model_epoch_36.pth.tar', type=str, metavar='N',
                    help='checkpoint of completeness score')

def main():
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"Testing G for concept completeness: {args.dataset}")
    if args.per_iter_completeness == "n":
        cal_completeness_stats(args)
    elif args.per_iter_completeness == "y":
        cal_completeness_stats_per_iter(args)



if __name__ == '__main__':
    main()
