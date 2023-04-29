#!/bin/sh
#SBATCH --output=path/cub_resnet_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_bb_train=cub_resnet_bb_train_$CURRENT.out
slurm_output_bb_test=cub_resnet_bb_test_$CURRENT.out
slurm_output_t_train=cub_resnet_t_train_$CURRENT.out
slurm_output_t_test=cub_resnet_t_test_$CURRENT.out
slurm_output_iter1_g_train=cub_resnet_iter1_g_train_$CURRENT.out
slurm_output_iter1_g_test=cub_resnet_iter1_g_test_$CURRENT.out
slurm_output_iter1_residual_train=cub_resnet_iter1_residual_train_$CURRENT.out
slurm_output_iter2_g_train=cub_resnet_iter2_g_train_$CURRENT.out
slurm_output_iter2_g_test=cub_resnet_iter2_g_test_$CURRENT.out
slurm_output_iter2_residual_train=cub_resnet_iter2_residual_train_$CURRENT.out
slurm_output_iter3_g_train=cub_resnet_iter3_g_train_$CURRENT.out
slurm_output_iter3_g_test=cub_resnet_iter3_g_test_$CURRENT.out
slurm_output_iter3_residual_train=cub_resnet_iter3_residual_train_$CURRENT.out
slurm_output_iter4_g_train=cub_resnet_iter4_g_train_$CURRENT.out
slurm_output_iter4_g_test=cub_resnet_iter4_g_test_$CURRENT.out
slurm_output_iter4_residual_train=cub_resnet_iter4_residual_train_$CURRENT.out
slurm_output_iter5_g_train=cub_resnet_iter5_g_train_$CURRENT.out
slurm_output_iter5_g_test=cub_resnet_iter5_g_test_$CURRENT.out
slurm_output_iter5_residual_train=cub_resnet_iter5_residual_train_$CURRENT.out
slurm_output_iter6_g_train=cub_resnet_iter6_g_train_$CURRENT.out
slurm_output_iter6_g_test=cub_resnet_iter6_g_test_$CURRENT.out
slurm_output_iter6_residual_train=cub_resnet_iter6_residual_train_$CURRENT.out
slurm_output_iter6_residual_test=cub_resnet_iter6_residual_test_$CURRENT.out
slurm_explanations=cub_resnet_explanations_$CURRENT.out

echo "CUB-200 ResNet101"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# BB model
# BB Training scripts

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_CUB.py --bs 16 --arch "ResNet101" > $slurm_output_bb_train


# BB Testing scripts
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_CUB.py --checkpoint-file "best_model_epoch_63.pth.tar" --save-activations True --layer "layer4" --bs 16 --arch "ResNet101"> $slurm_output_bb_test


# T model
# train
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_CUB.py --checkpoint-file "best_model_epoch_63.pth.tar" --bs 32 --layer "layer4" --flattening-type "adaptive" --arch "ResNet101" > $slurm_output_t_train

# Test
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_CUB.py --checkpoint-file "best_model_epoch_63.pth.tar" --checkpoint-file-t "best_model_epoch_62.pth.tar" --save-concepts True --bs 16 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet101"> $slurm_output_t_test


# MoIE Training scripts

#---------------------------------
# # iter 1
#---------------------------------

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter1_g_train


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 1 --expert-to-train "explainer" --dataset "cub" --cov 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101">  $slurm_output_iter1_g_test


python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 1 --expert-to-train "residual" --dataset "cub" --cov 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"> $slurm_output_iter1_residual_train




#---------------------------------
# # iter 2
#---------------------------------
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 2 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" >  $slurm_output_iter2_g_train


# # Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 2 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter2_g_test


# # # Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 2 --expert-to-train "residual" --dataset "cub" --cov 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter2_residual_train


#---------------------------------
# # iter 3
#---------------------------------
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 3 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter3_g_train


# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 3 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01  --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter3_g_test


# # # # Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 3 --expert-to-train "residual" --dataset "cub" --cov 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter3_residual_train


#---------------------------------
# # iter 4
#---------------------------------
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 4 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter4_g_train


# # # Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 4 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter4_g_test


# # # # # Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 4 --expert-to-train "residual" --dataset "cub" --cov 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter4_residual_train


# #---------------------------------
# # # iter 5
# #---------------------------------
# # Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 5 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter5_g_train


# # # # Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar"  --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 5 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter5_g_test


# # # # # # Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 5 --expert-to-train "residual" --dataset "cub" --cov 0.2 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter5_residual_train



# # #---------------------------------
# # # # iter 6
# # #---------------------------------
# # # Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter5" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 6 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter6_g_train


# # # # Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter5" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 6 --expert-to-train "explainer" --dataset "cub" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter6_g_test


# # # # # # Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter5" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 6 --expert-to-train "residual" --dataset "cub" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter5_residual_train

# # # # # # Train final residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_CUB.py --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter5" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "best_model_epoch_63.pth.tar" --iter 6 --expert-to-train "residual" --dataset "cub" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter6_residual_train



# # #---------------------------------
# # # # Explanations
# # #---------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/FOLs_vision_main.py --arch "ResNet101" --dataset "cub" --iterations 6  > $slurm_explanations
