import argparse
import os
import sys

from BB.experiments_BB_CUB import test

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))

parser = argparse.ArgumentParser(description='CUB Training')
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
parser.add_argument('--checkpoint-file', metavar='file',
                    default='best_model_epoch_63.pth.tar',
                    help='checkpoint file of BB')
parser.add_argument('--output', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                    help='path to output logs')
parser.add_argument('--attribute-file-name', metavar='file',
                    default='attributes.npy',
                    help='file containing all the concept attributes')
parser.add_argument('--save-activations', type=bool, default=True, help='test BB or save activation maps of BB')
parser.add_argument('--layer', type=str, default="layer4", help='batch size for training of t')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="cub", help='dataset name')
parser.add_argument('--img-size', type=int, default=448, help='image\'s size for transforms')
parser.add_argument('--bs', '--batch-size', default=1, type=int, metavar='N', help='batch size BB')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epochs', type=int, default=95, help='batch size for training')
parser.add_argument('--arch', type=str, default="ResNet50", required=True, help='ResNet50 or ResNet101 or ResNet152')
parser.add_argument("--num_steps", default=10000, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
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
                             'has_wing_pattern_multicolored'])
parser.add_argument('--labels', nargs='+',
                    default=['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani',
                             'Crested_Auklet',
                             'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird',
                             'Red_winged_Blackbird',
                             'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting',
                             'Lazuli_Bunting',
                             'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat',
                             'Eastern_Towhee',
                             'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant',
                             'Bronzed_Cowbird',
                             'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo',
                             'Mangrove_Cuckoo',
                             'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker',
                             'Acadian_Flycatcher',
                             'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher',
                             'Scissor_tailed_Flycatcher',
                             'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar',
                             'Gadwall',
                             'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe',
                             'Horned_Grebe',
                             'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak',
                             'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull',
                             'Heermann_Gull',
                             'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull',
                             'Anna_Hummingbird',
                             'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger',
                             'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco',
                             'Tropical_Kingbird',
                             'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher',
                             'Ringed_Kingfisher',
                             'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon',
                             'Mallard',
                             'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird',
                             'Nighthawk',
                             'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole',
                             'Orchard_Oriole',
                             'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee',
                             'Sayornis',
                             'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven',
                             'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike',
                             'Baird_Sparrow',
                             'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow',
                             'House_Sparrow',
                             'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow',
                             'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow',
                             'Seaside_Sparrow',
                             'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow',
                             'White_throated_Sparrow',
                             'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow',
                             'Scarlet_Tanager',
                             'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern',
                             'Elegant_Tern', 'Forsters_Tern',
                             'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher',
                             'Black_capped_Vireo',
                             'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo',
                             'White_eyed_Vireo',
                             'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler',
                             'Black_throated_Blue_Warbler',
                             'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler',
                             'Chestnut_sided_Warbler',
                             'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler',
                             'Mourning_Warbler',
                             'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler',
                             'Pine_Warbler',
                             'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler',
                             'Wilson_Warbler',
                             'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush',
                             'Bohemian_Waxwing',
                             'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker',
                             'Red_bellied_Woodpecker',
                             'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren',
                             'Cactus_Wren',
                             'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren',
                             'Common_Yellowthroat'])

parser.add_argument('--spurious-specific-classes', type=str, default="n", required=False, help='y or n')
parser.add_argument('--spurious-waterbird-landbird', type=str, default="n", required=False, help='y or n')
parser.add_argument('--projected', type=str, default="n", required=False, help='n')

def main():
    args = parser.parse_args()
    print("Inputs")
    if args.spurious_specific_classes == "y":
        print("Spurious specific classes")
        args.data_root = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/spurious/CUB_200_2011"
        args.logs = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log/spurious-cub-specific-classes"
        args.checkpoints = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/spurious-cub-specific-classes"
        args.output = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/spurious-cub-specific-classes"
        args.attribute_file_name = "attributes_spurious.npy"
        args.concept_names.extend(['has_water', 'has_land'])
    elif args.spurious_waterbird_landbird == "y":
        print("Spurious whole dataset")
        args.data_root = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/spurious/waterbird_complete95_forest2water2"
        args.logs = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log/spurious-cub-waterbird-landbird-{args.img_size}"
        args.checkpoints = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/spurious-cub-waterbird-landbird-{args.img_size}"
        args.output = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/spurious-cub-waterbird-landbird-{args.img_size}"
        args.attribute_file_name = "attributes_spurious.npy"
        args.concept_names.extend(["has_ocean", "has_lake", "has_bamboo", "has_forest"])

        args.lr = 0.001
        args.epochs = 300
        args.weight_decay = 0.0001
        args.labels = ["0 (Landbird)", "1 (Waterbird)"]
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    test(args)
    # if args.save_activations:
    #     save_activations(args)


if __name__ == '__main__':
    main()
