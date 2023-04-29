import argparse
import os
import sys

from BB.experiments_BB_awa2 import train

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))

parser = argparse.ArgumentParser(description='Awa2 Training')
parser.add_argument('--data-root', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/awa2',
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
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="awa2", help='dataset name')
parser.add_argument('--img-size', type=int, default=224, help='image\'s size for transforms')
parser.add_argument(
    '--bs', '--batch-size', '--train_batch_size', '--eval_batch_size', default=32, type=int,
    metavar='N', help='batch size BB'
)
parser.add_argument("--eval_every", default=100, type=int,
                    help="100Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")
parser.add_argument('--lr', '--learning-rate', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight-decay', type=float, default=0, help='weight_decay for SGD')
parser.add_argument('--epochs', type=int, default=95, help='batch size for training')
parser.add_argument('--arch', type=str, default="ResNet50", required=True, help='ResNet50 or ResNet101 or ResNet152')

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
parser.add_argument('--profile', default='n', type=str, metavar='N', help='run_profiler')


def main():
    print("Train BB for AWA2 dataset")
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    if args.arch == "ResNet50" or args.arch == "ResNet101":
        train(args)


if __name__ == '__main__':
    main()
