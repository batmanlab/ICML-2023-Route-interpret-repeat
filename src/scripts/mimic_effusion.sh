#!/bin/sh
#SBATCH --output=path/effusion_mimic_%j.out
pwd
hostname
date
CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_bb_train=effusion_mimic_bb_train_$CURRENT.out
slurm_output_bb_test=effusion_mimic_bb_test_$CURRENT.out
slurm_output_t_train=effusion_mimic_t_train_$CURRENT.out
slurm_output_t_test=effusion_mimic_t_test_$CURRENT.out
slurm_output_iter1_g_train=effusion_mimic_iter1_g_train_$CURRENT.out
slurm_output_iter1_g_test=effusion_mimic_iter1_g_test_$CURRENT.out
slurm_output_iter1_residual_train=effusion_mimic_iter1_residual_train_$CURRENT.out
slurm_output_iter1_residual_test=effusion_mimic_iter1_residual_test_$CURRENT.out
slurm_output_iter2_g_train=effusion_mimic_iter2_g_train_$CURRENT.out
slurm_output_iter2_g_test=effusion_mimic_iter2_g_test_$CURRENT.out
slurm_output_iter2_residual_train=effusion_mimic_iter2_residual_train_$CURRENT.out
slurm_output_iter2_residual_test=effusion_mimic_iter2_residual_test_$CURRENT.out
slurm_output_iter3_g_train=effusion_mimic_iter3_g_train_$CURRENT.out
slurm_output_iter3_g_test=effusion_mimic_iter3_g_test_$CURRENT.out
slurm_output_iter3_residual_train=effusion_mimic_iter3_residual_train_$CURRENT.out
slurm_output_iter3_residual_test=effusion_mimic_iter3_residual_test_$CURRENT.out

slurm_performance_all=effusion_mimic_performance_$CURRENT.out

echo "Effusion MIMIC-CXR"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

#######################################
# Effusion
#######################################

## BB Training
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_BB_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --resize=512 \
  --resume='' \
  --loss="CE" \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_bb_train

## BB Testing
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_BB_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --resize=512 \
  --resume='' \
  --loss="CE" \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --checkpoint-bb="g_best_model_epoch_8.pth.tar" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_bb_test

# T Training
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W" \
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb="g_best_model_epoch_8.pth.tar" \
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_t_train

# T Testing
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W" \
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb="g_best_model_epoch_8.pth.tar" \
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_10.pth.tar" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_t_test

#######################################
# iter1
#######################################
# train g (Epoch 193)
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
  --iter 1 \
  --icml "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.5 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter1_g_train

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
  --iter 1 \
  --icml "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.5 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter1_g_test

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
  --iter 1 \
  --icml "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.5 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter1_residual_train

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
  --iter 1 \
  --icml "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.5 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --arch "densenet121" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter1_residual_test

########################
# iter 2
########################
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
  --iter 2 \
  --expert-to-train "explainer" \
  --icml "y" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter2_g_train

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
  --iter 2 \
  --expert-to-train "explainer" \
  --icml "y" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" "model_seq_epoch_113.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter2_g_test

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
  --iter 2 \
  --expert-to-train "residual" \
  --icml "y" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" "model_seq_epoch_113.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter2_residual_train

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
  --iter 2 \
  --expert-to-train "residual" \
  --icml "y" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --arch "densenet121" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" "model_seq_epoch_113.pth.tar" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"  >$slurm_output_iter2_residual_test

########################
# iter 3
########################
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
  --iter 3 \
  --icml "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.05 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" "model_seq_epoch_113.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter3_g_train

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
  --iter 3 \
  --icml "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.05 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" "model_seq_epoch_113.pth.tar" "model_seq_epoch_107.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter3_g_test

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_mimic_cxr.py \
  --iter 3 \
  --icml "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.05 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" "model_seq_epoch_113.pth.tar" "model_seq_epoch_107.pth.tar" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)" >$slurm_output_iter3_residual_train

python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_mimic_cxr.py \
  --iter 3 \
  --icml "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.05 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 96.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --prev_chk_pt_explainer_folder "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.5_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_123.pth.tar" "model_seq_epoch_113.pth.tar" "model_seq_epoch_107.pth.tar" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --selected-obs="effusion" \
  --labels "0 (No Effusion)" "1 (Effusion)"  >$slurm_output_iter3_residual_test

# All performance
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/performance_calculation_mimic_cxr_main.py --iterations 3 --icml "y" --disease "effusion" --model "MoIE" >$slurm_performance_all
