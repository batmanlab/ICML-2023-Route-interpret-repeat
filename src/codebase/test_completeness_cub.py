import argparse
import os
import sys

from Explainer.completeness_score import cal_completeness_stats, cal_completeness_stats_per_iter

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
parser.add_argument('--bs', '--batch-size', default=16, type=int, metavar='N', help='batch size BB')
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
parser.add_argument('--hidden-nodes', default=10, type=int, help='hidden nodes of the explainer model')
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
    print("Inputs")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"Testing G for concept completeness: {args.dataset}")
    if args.per_iter_completeness == "n":
        cal_completeness_stats(args)
    elif args.per_iter_completeness == "y":
        cal_completeness_stats_per_iter(args)


if __name__ == '__main__':
    main()
