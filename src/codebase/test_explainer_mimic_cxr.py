import argparse
import os
import sys
from Explainer.experiments_explainer_mimic_cxr_icml import test_glt_icml, get_FOL_icml

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))

parser = argparse.ArgumentParser(description='mimic_cxr Training')
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

parser.add_argument('--resize', default=512, type=int,
                    help='input image resize')
parser.add_argument('--crop', default=448, type=int,
                    help='resize image crop')
parser.add_argument('--degree', default=10, type=int,
                    help='rotation range [-degree, +degree].')
parser.add_argument('--mini-data', default=None, type=int, help='small dataset for debugging')

parser.add_argument('--logs', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log',
                    help='path to tensorboard logs')
parser.add_argument('--checkpoints', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                    help='path to checkpoints')
parser.add_argument('--checkpoint-model', metavar='file', nargs="+",
                    default=['model_seq_epoch_193.pth.tar'],
                    help='checkpoint files all the experts including the current iteration. For example: if the current iteration is 3, include the checkpoint files expert 1, expert 2 and expert 3')
parser.add_argument('--checkpoint-residual', metavar='file', nargs="+",
                    default=['model_residual_best_model_epoch_2.pth.tar'],
                    help='checkpoint files all the residuals including the current iteration. For example: if the current iteration is 3, include the checkpoint files residual 1, residual 2 and residual 3')
parser.add_argument('--output', metavar='DIR',
                    default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                    help='path to output logs')

parser.add_argument('--iter', default=1, type=int, metavar='N', help='seed')
parser.add_argument('--pool1', metavar='ARCH', default='average',
                    help='type of pooling layer for net1. the options are: average, max, log-sum-exp')
parser.add_argument('--expert-to-train', default="explainer", type=str, metavar='N',
                    help='which expert to train? explainer or residual')
parser.add_argument('--seed', default=1, type=int, metavar='N', help='seed')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="mimic_cxr", help='dataset name')
parser.add_argument('--img-size', type=int, default=448, help='image\'s size for transforms')
parser.add_argument('--cov', default=0.45, type=float, help='coverage of the dataset')
parser.add_argument('--alpha', default=0.5, type=float, help='trade off for Aux explainer using Selection Net')
parser.add_argument('--selection-threshold', default=0.5, type=float,
                    help='selection threshold of the selector for the test/val set')
parser.add_argument('--bs', '--batch-size', default=16, type=int, metavar='N', help='batch size BB')
parser.add_argument('--dataset-folder-concepts', type=str,
                    default="lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4",
                    help='dataset folder of concept bank')
parser.add_argument('--lr-residual', '--learning-rate-residual', default=0.001, type=float,
                    metavar='LR', help='initial learning rate of bb residual')
parser.add_argument('--momentum-residual', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight-decay-residual', type=float, default=1e-4, help='weight_decay for SGD')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--input-size-pi', default=2048, type=int,
                    help='input size of pi - 2048 for layer4 (ResNet) or 1024 for layer3 (ResNet) ')
parser.add_argument('--temperature-lens', default=0.7, type=float, help='temperature for entropy layer in lens')
parser.add_argument('--lambda-lens', default=0.0001, type=float, help='weight for entropy loss')
parser.add_argument('--alpha-KD', default=0.9, type=float, help='weight for KD loss by Hinton')
parser.add_argument('--lm', default=32.0, type=float, help='lagrange multiplier for selective KD loss')
parser.add_argument('--temperature-KD', default=10, type=float, help='temperature for KD loss')
parser.add_argument('--conceptizator', default='identity_bool', type=str, help='activation')
parser.add_argument('--hidden-nodes', nargs="+", default=[10], type=int, help='hidden nodes of the explainer model')
parser.add_argument('--explainer-init', default=None, type=str, help='Initialization of explainer')
parser.add_argument('--epochs', type=int, default=500, help='batch size for training the explainer - g')
parser.add_argument('--epochs-residual', type=int, default=50, help='batch size for training the residual')
parser.add_argument('--arch', type=str, default="densenet121", help='densenet121')
parser.add_argument('--optim', type=str, default="SGD", help='optimizer of GLT')
parser.add_argument('--layer', type=str, default="layer4", help='batch size for training of t')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
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
parser.add_argument('--landmark-names-unspec', nargs='+',
                    default=['cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])

parser.add_argument('--full-obs', nargs='+',
                    default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                             'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                             'process', 'abnormality', 'enlarge', 'tip', 'low',
                             'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                             'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                             'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid',
                             'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                             'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate',
                             'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
                             'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                             'tail_abnorm_obs', 'excluded_obs'])
parser.add_argument('--abnorm-obs', nargs='+',
                    default=['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                             'process', 'abnormality', 'enlarge', 'tip', 'low',
                             'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                             'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                             'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid',
                             'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                             'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd',
                             'infiltrate', 'obscure', 'deformity', 'hernia',
                             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
                             'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
parser.add_argument('--norm-obs', nargs='+',
                    default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                             'expand', 'hyperinflate'])
parser.add_argument('--tail-abnorm-obs', nargs='+', default=['tail_abnorm_obs'])
parser.add_argument('--excluded-obs', nargs='+', default=['excluded_obs'])
# PNU labels
parser.add_argument('--landmark_label', default='PN',
                    help='anatomical landmark label type, PN or PUN.')
parser.add_argument('--obs-u-alpha-method', default='discard',
                    help='how to deal with top alpha U samples? discard or replace (-1 to 1)')
parser.add_argument('--warm-up', default=0, type=int,
                    help='number of epochs warm up.')
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
                             'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium', 'right_ventricle',
                             'aorta', 'svc',
                             'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
parser.add_argument('--abnorm-obs-concepts', nargs='+',
                    default=['effusion', 'opacity', 'edema', 'atelectasis', 'tube', 'consolidation',
                             'process', 'abnormality', 'enlarge', 'tip', 'low',
                             'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                             'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                             'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid',
                             'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                             'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd',
                             'infiltrate', 'obscure', 'deformity', 'hernia',
                             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
                             'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
parser.add_argument('--labels', nargs='+',
                    default=['0 (No Pneumothorax)', '1 (Pneumothorax)'])
parser.add_argument('--selected-obs', nargs='+', default=['pneumothorax'])
parser.add_argument('--bb-chkpt-folder', type=str,
                    default="lr_0.01_epochs_60_loss_CE",
                    help='dataset folder of concepts')
parser.add_argument('--metric', type=str,
                    default="auroc",
                    help='auroc/recall')
parser.add_argument('--checkpoint-bb', metavar='file', default='g_best_model_epoch_4.pth.tar',
                    help='checkpoint file of BB')
parser.add_argument('--prev_chk_pt_explainer_folder', nargs='+', type=str,
                    default="densenet121_lr_0.1_SGD_temperature-lens_7.6_cov_0.1_alpha_0.5_selection-threshold_0.5_lm_32.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4",
                    help='chkpt explainer')
parser.add_argument('--soft', default='y', type=str, metavar='N', help='soft/hard concept?')
parser.add_argument('--with_seed', default='n', type=str, metavar='N', help='trying diff seeds for paper')
parser.add_argument('--disease', type=str, default="effusion", help='dataset name')
parser.add_argument('--icml', default='n', type=str, metavar='N', help='for icml or miccai')


def main():
    args = parser.parse_args()
    print("Inputs")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    test_glt_icml(args)
    get_FOL_icml(args)


if __name__ == '__main__':
    main()
