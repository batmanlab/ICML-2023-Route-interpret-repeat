#!/bin/sh
#SBATCH --output=path/test_time_interventions_difficult_expert_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_cub_vit=cub_vit_$CURRENT.out
slurm_output_cub_resnet=cub_resnet_$CURRENT.out
slurm_output_awa2_vit=awa2_vit_$CURRENT.out
slurm_output_awa2_resnet=awa2_resnet_$CURRENT.out

echo "Test Time Interventions for difficult samples covered by later experts"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000


# -----------------------------------------------------
# CUB_VIT
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 5 6 --model "MoIE" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 90 108 > $slurm_output_cub_vit

# -----------------------------------------------------
# CUB_ResNet
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 5 6 --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 90 108 > $slurm_output_cub_resnet

# -----------------------------------------------------
# Awa2_vit
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 5 6 --model "MoIE" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85 > $slurm_output_awa2_vit

# -----------------------------------------------------
# Awa2_Resnet
# -----------------------------------------------------
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 3 4 --model "MoIE" --arch "ResNet101" --dataset "awa2" --iterations 4 --top_K 3 5 10 15 20 25 30 50 75 85 > $slurm_output_awa2_resnet
