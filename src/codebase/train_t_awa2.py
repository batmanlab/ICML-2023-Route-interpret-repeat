import argparse
import os
import sys

from BB.experiments_t_awa2 import train_t

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
parser.add_argument('--checkpoint-file', metavar='file',
                    default='best_model_epoch_63.pth.tar',
                    help='path to checkpoints')
parser.add_argument('--output', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                    help='path to output logs')
parser.add_argument('--attribute-file-name', metavar='file',
                    default='attributes.npy',
                    help='file containing all the concept attributes')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="awa2", help='dataset name')
parser.add_argument('--img-size', type=int, default=448, help='image\'s size for transforms')
parser.add_argument('--bs', '--batch-size', default=16, type=int, metavar='N', help='batch size BB')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epochs', type=int, default=95, help='batch size for training')
parser.add_argument('--solver-LR', type=str, default="sgd", help='solver - sgd/adam')
parser.add_argument('--loss-LR', type=str, default="BCE", help='loss - focal/BCE')
parser.add_argument('--epochs-LR', type=int, default=200, help='epoch of training t')
parser.add_argument('--layer', type=str, default="layer4", help='layer of bb to be used as phi')
parser.add_argument('--flattening-type', type=str, default="adaptive", help='flatten or adaptive or maxpool')
parser.add_argument('--smoothing_value', type=float, default=0.0,
                    help="Label smoothing value\n")
parser.add_argument('--arch', type=str, default="ResNet101", required=True, help='ResNet50 or ResNet101 or ResNet152')
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
parser.add_argument('--projected', type=str, default="n", required=False, help='n')

def main():
    print("Getting T ready...")
    args = parser.parse_args()
    print("Inputs")


    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    train_t(args)


if __name__ == '__main__':
    main()
