#!/bin/sh
#SBATCH --output=path/awa2_resnet_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_bb_train=awa2_resnet_bb_train_$CURRENT.out
slurm_output_bb_test=awa2_resnet_bb_test_$CURRENT.out
slurm_output_t_train=awa2_resnet_t_train_$CURRENT.out
slurm_output_t_test=awa2_resnet_t_test_$CURRENT.out
slurm_output_iter1_g_train=awa2_resnet_iter1_g_train_$CURRENT.out
slurm_output_iter1_g_test=awa2_resnet_iter1_g_test_$CURRENT.out
slurm_output_iter1_residual_train=awa2_resnet_iter1_residual_train_$CURRENT.out
slurm_output_iter2_g_train=awa2_resnet_iter2_g_train_$CURRENT.out
slurm_output_iter2_g_test=awa2_resnet_iter2_g_test_$CURRENT.out
slurm_output_iter2_residual_train=awa2_resnet_iter2_residual_train_$CURRENT.out
slurm_output_iter3_g_train=awa2_resnet_iter3_g_train_$CURRENT.out
slurm_output_iter3_g_test=awa2_resnet_iter3_g_test_$CURRENT.out
slurm_output_iter3_residual_train=awa2_resnet_iter3_residual_train_$CURRENT.out
slurm_output_iter4_g_train=awa2_resnet_iter4_g_train_$CURRENT.out
slurm_output_iter4_g_test=awa2_resnet_iter4_g_test_$CURRENT.out
slurm_output_iter4_residual_train=awa2_resnet_iter4_residual_train_$CURRENT.out

slurm_explanations=awa2_resnet_explanations_$CURRENT.out

echo "awa2 ResNet101"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# BB model
# BB Training scripts

python ./codebase/train_BB_awa2.py --bs 16 --arch "ResNet101" > $slurm_output_bb_train


# T model 
# train
python ../codebase/train_t_awa2.py --checkpoint-file "g_best_model_epoch_[epoch].pth.tar" --bs 32 --layer "layer4" --flattening-type "adaptive" --arch "ResNet101" > $slurm_output_t_train

# Test
python ../codebase/test_t_awa2.py --checkpoint-file "g_best_model_epoch_[epoch].pth.tar" --checkpoint-file-t "g_best_model_epoch_199.pth.tar" --save-concepts True --bs 16 --solver-LR "sgd" --loss-LR "BCE" --layer "layer4" --flattening-type "adaptive" --arch "ResNet101"> $slurm_output_t_test


# MoIE Training scripts

#---------------------------------
# # iter 1 
#---------------------------------

python ../codebase/train_explainer_awa2.py --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 1 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter1_g_train


python ../codebase/test_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 1 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101">  $slurm_output_iter1_g_test


python ../codebase/train_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 1 --expert-to-train "residual" --dataset "awa2" --cov 0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"> $slurm_output_iter1_residual_train




#---------------------------------
# # iter 2 
#---------------------------------
# Train explainer
python ../codebase/train_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 2 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" >  $slurm_output_iter2_g_train


# # Test explainer
python ../codebase/test_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 2 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter2_g_test


# # # Train residual
python ../codebase/train_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 2 --expert-to-train "residual" --dataset "awa2" --cov 0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter2_residual_train


#---------------------------------
# # iter 3 
#---------------------------------
# Train explainer
python ../codebase/train_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar"  "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter2" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 3 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter3_g_train


# Test explainer
python ../codebase/test_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter2" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 3 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01  --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter3_g_test


# # # # Train residual
python ../codebase/train_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar"  --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter2" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 3 --expert-to-train "residual" --dataset "awa2" --cov 0.4  0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter3_residual_train


#---------------------------------
# # iter 4
#---------------------------------
# Train explainer
python ../codebase/train_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar"  "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter3" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 4 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101"  > $slurm_output_iter4_g_train


# # # Test explainer
python ../codebase/test_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter3" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 4 --expert-to-train "explainer" --dataset "awa2" --cov 0.4  0.4  0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter4_g_test


# # # # # Train residual
python ../codebase/train_explainer_awa2.py --checkpoint-model "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" "model_g_best_model.pth.tar" --checkpoint-residual "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" "model_residual_best_model.pth.tar" --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/awa2/explainer/ResNet101/lr_0.001_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.4_lr_0.001/iter3" --root-bb "lr_0.001_epochs_95" --checkpoint-bb "g_best_model_epoch_[epoch].pth.tar" --iter 4 --expert-to-train "residual" --dataset "awa2" --cov 0.4  0.4  0.4  0.4  --bs 16 --dataset-folder-concepts "lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE" --lr 0.01 0.01 0.01 0.01 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "layer4" --arch "ResNet101" > $slurm_output_iter4_residual_train


# # #---------------------------------
# # # # Explanations
# # #---------------------------------
# Update ../codebase/Completeness_and_interventions/paths_MoIE.json file with appropriate paths for the checkpoints and outputs
python ../codebase/FOLs_vision_main.py --arch "ResNet101" --dataset "awa2" --iterations 4 > $slurm_explanations