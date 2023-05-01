#!/bin/sh
#SBATCH --output=path/completeness_scores_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_cub_vit_concept_mask=cub_vit_concept_mask_$CURRENT.out
slurm_output_cub_vit_concept_completeness=cub_vit_concept_completeness_$CURRENT.out
slurm_output_cub_resnet_concept_mask=cub_resnet_concept_mask_$CURRENT.out
slurm_output_cub_resnet_concept_completeness=cub_resnet_concept_completeness_$CURRENT.out
slurm_output_ham_concept_mask=ham_concept_mask_$CURRENT.out
slurm_output_ham_concept_completeness=ham_concept_completeness_$CURRENT.out
slurm_output_awa2_concept_mask=awa2_concept_mask_$CURRENT.out
slurm_output_awa2_concept_completeness=awa2_concept_completeness_$CURRENT.out

echo "Completeness Scores"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# -----------------------------------------------------
# CUB_VIT
# -----------------------------------------------------
# MoIE
python ./src/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_vit_concept_mask

python ./src/codebase/concept_completeness_main.py --model "MoIE" --epochs 3 --arch "ViT-B_16" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_vit_concept_completeness



# -----------------------------------------------------
# CUB_ResNet101
# -----------------------------------------------------
# MoIE

python ./src/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_resnet_concept_mask

python ./src/codebase/concept_completeness_main.py --model "MoIE" --epochs 75 --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_resnet_concept_completeness


# -----------------------------------------------------
# HAM10k
# -----------------------------------------------------
# MoIE

python ./src/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 7 8 > $slurm_output_ham_concept_mask

python ./src/codebase/concept_completeness_main.py --model "MoIE" --epochs 10 --arch "Inception_V3" --dataset "HAM10k" --iterations 6 --top_K 1 2 3 4 5 6 7 8 > $slurm_output_ham_concept_completeness



# -----------------------------------------------------
# Awa2_VIT
# -----------------------------------------------------
# MoIE
python ./src/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85 > $slurm_output_awa2_concept_mask

python ./src/codebase/concept_completeness_main.py --model "MoIE" --epochs 10 --arch "ViT-B_16" --dataset "awa2" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 85 > $slurm_output_awa2_concept_completeness

