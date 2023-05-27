#!/bin/sh
#SBATCH --output=path/ham_inception_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_bb_train=ham_inception_bb_train_$CURRENT.out
slurm_output_bb_test=ham_inception_bb_test_$CURRENT.out
slurm_output_t_train=ham_inception_t_train_$CURRENT.out
slurm_output_t_test=ham_inception_t_test_$CURRENT.out
slurm_output_iter1_g_train=ham_inception_iter1_g_train_$CURRENT.out
slurm_output_iter1_g_test=ham_inception_iter1_g_test_$CURRENT.out
slurm_output_iter1_residual_train=ham_inception_iter1_residual_train_$CURRENT.out
slurm_output_iter1_residual_test=ham_inception_iter1_residual_test_$CURRENT.out
slurm_output_iter2_g_train=ham_inception_iter2_g_train_$CURRENT.out
slurm_output_iter2_g_test=ham_inception_iter2_g_test_$CURRENT.out
slurm_output_iter2_residual_train=ham_inception_iter2_residual_train_$CURRENT.out
slurm_output_iter2_residual_test=ham_inception_iter2_residual_test_$CURRENT.out
slurm_output_iter3_g_train=ham_inception_iter3_g_train_$CURRENT.out
slurm_output_iter3_g_test=ham_inception_iter3_g_test_$CURRENT.out
slurm_output_iter3_residual_train=ham_inception_iter3_residual_train_$CURRENT.out
slurm_output_iter3_residual_test=ham_inception_iter3_residual_test_$CURRENT.out
slurm_output_iter4_g_train=ham_inception_iter4_g_train_$CURRENT.out
slurm_output_iter4_g_test=ham_inception_iter4_g_test_$CURRENT.out
slurm_output_iter4_residual_train=ham_inception_iter4_residual_train_$CURRENT.out
slurm_output_iter4_residual_test=ham_inception_iter4_residual_test_$CURRENT.out
slurm_output_iter5_g_train=ham_inception_iter5_g_train_$CURRENT.out
slurm_output_iter5_g_test=ham_inception_iter5_g_test_$CURRENT.out
slurm_output_iter5_residual_train=ham_inception_iter5_residual_train_$CURRENT.out
slurm_output_iter5_residual_test=ham_inception_iter5_residual_test_$CURRENT.out
slurm_output_iter6_g_train=ham_inception_iter6_g_train_$CURRENT.out
slurm_output_iter6_g_test=ham_inception_iter6_g_test_$CURRENT.out
slurm_output_iter6_residual_train=ham_inception_iter6_residual_train_$CURRENT.out
slurm_output_iter6_residual_test=ham_inception_iter6_residual_test_$CURRENT.out
slurm_explanations=ham_inception_explanations_$CURRENT.out

echo "HAM10k Inception_V3"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7

#################################################
# Instructions for downloading the BB model
# Get the BB model from the Posthoc Concept Bottleneck repo (https://github.com/mertyg/post-hoc-cbm)
# or Get the checkpoints directly from https://drive.google.com/drive/folders/1WscikgfyQWg1OTPem_JZ-8EjbCQ_FHxm
# Do not change the name ham10000.pth
#################################################

# T model 
# train
python ../codebase/train_t_ham10k.py  --bs 32 --arch "Inception_V3" > $slurm_output_t_train


# MoIE Training scripts
#---------------------------------
# # iter 1
#---------------------------------
# Train explainer
python ../codebase/train_explainer_ham10k.py --iter 1 --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter1_g_train

# Test explainer
python ../codebase/test_explainer_ham10k.py --iter 1 --checkpoint-model "model_g_best_model_epoch_2.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" >  $slurm_output_iter1_g_test

# Train residual
python ../codebase/train_explainer_ham10k.py --iter 1 --checkpoint-model "model_g_best_model_epoch_2.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter1_residual_train

# Test residual
python ../codebase/test_explainer_ham10k.py --iter 1 --checkpoint-model "model_g_best_model_epoch_2.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 --bs 32 --lr 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter1_residual_test


#---------------------------------
# # iter 2
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python ../codebase/train_explainer_ham10k.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_2.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter2_g_train

# Test explainer
python ../codebase/test_explainer_ham10k.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_2.pth.tar" "model_g_best_model_epoch_342.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" >  $slurm_output_iter2_g_test



# Train residual
python ../codebase/train_explainer_ham10k.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_2.pth.tar" "model_g_best_model_epoch_342.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter2_residual_train


# Test residual
python ../codebase/test_explainer_ham10k.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_2.pth.tar" "model_g_best_model_epoch_342.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 --bs 32 --lr 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter2_residual_test



#---------------------------------
# # iter 3
#---------------------------------
# Train explainer
python ../codebase/train_explainer_ham10k.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_2.pth.tar" "model_g_best_model_epoch_342.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar"  --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter3_g_train

# Test explainer
python ../codebase/test_explainer_ham10k.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_2.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" >  $slurm_output_iter3_g_test


# Train residual
python ../codebase/train_explainer_ham10k.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar"  --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter3_residual_train

# Test residual
python ../codebase/test_explainer_ham10k.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar"  "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter3_residual_test




#---------------------------------
# # iter 4
#---------------------------------
# Train explainer
python ../codebase/train_explainer_ham10k.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar"  "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter4_g_train

# Test explainer
python ../codebase/test_explainer_ham10k.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter4_g_test


# Train residual
python ../codebase/train_explainer_ham10k.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter4_residual_train


# Test residual
python ../codebase/test_explainer_ham10k.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_5.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter4_residual_test


#---------------------------------
# # iter 5
#---------------------------------
# Train explainer
python ../codebase/train_explainer_ham10k.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_5.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter5_g_train

# Test explainer
python ../codebase/test_explainer_ham10k.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" "model_g_best_model_epoch_140.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_5.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" >  $slurm_output_iter5_g_test


# Train residual
python ../codebase/train_explainer_ham10k.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" "model_g_best_model_epoch_140.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_5.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter5_residual_train


# Test residual
python ../codebase/test_explainer_ham10k.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" "model_g_best_model_epoch_140.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_5.pth.tar" "model_residual_best_model_epoch_13.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter5_residual_test


#---------------------------------
# # iter 6
#---------------------------------
# Train explainer
python ../codebase/train_explainer_ham10k.py --iter 6 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter5" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" "model_g_best_model_epoch_140.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_5.pth.tar" "model_residual_best_model_epoch_13.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" > $slurm_output_iter6_g_train


# Test explainer
python ../codebase/test_explainer_ham10k.py --iter 6 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter4" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.2/iter5" --checkpoint-model "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_342.pth.tar" "model_g_best_model_epoch_128.pth.tar" "model_g_best_model_epoch_442.pth.tar" "model_g_best_model_epoch_140.pth.tar" "model_g_best_model_epoch_142.pth.tar" --checkpoint-residual "model_residual_best_model_epoch_9.pth.tar" "model_residual_best_model_epoch_4.pth.tar" "model_residual_best_model_epoch_3.pth.tar" "model_residual_best_model_epoch_5.pth.tar" "model_residual_best_model_epoch_13.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.2 0.2 0.2 0.2 0.2 0.2 --bs 32 --lr 0.01 0.01 0.01 0.01 0.01 0.01 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3" >  $slurm_output_iter6_g_test


# # #---------------------------------
# # # # Explanations
# # #---------------------------------
# Update ../codebase/Completeness_and_interventions/paths_MoIE.json file with appropriate paths for the checkpoints and outputs
python ../codebase/FOLs_vision_main.py --arch "Inception_V3" --dataset "HAM10k" --iterations 6 > $slurm_explanations