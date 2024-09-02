#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=40:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N endothelium

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module load conda
cd /grand/GeomicVar/for_bick_group/
conda activate geneformer_env_B/

cd /grand/GeomicVar/pershy1/scripts

python geneformer_isp_finetunedclassifier_yp_012724.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/vascular2024_endothelium/vascular2024_endothelium.dataset --filter_field cell_type --filter_class "Endothelial cells" --model /grand/GeomicVar/pershy1/classifiers/240418_geneformer_CellClassifier_chip_vs_control_endothelium_L2048_B6_LR5e-05_LSlinear_WU500_E10_Oadamw_F0 --state_key chip --start_state "CHIP"  --goal_state "Control" --embeddings_output_dir /grand/GeomicVar/pershy1/results/embeddings/vascular2024_control_chip_endothelium_finetuned --state_embs_prefix vascular2024_control_chip_endothelium --perturb_output /grand/GeomicVar/pershy1/results/isp/vascular2024_control_chip_endothelium_perturb_output_12L --perturb_prefix vascular2024_control_chip_endothelium_perturb_output_12L --perturb_stats_output /grand/GeomicVar/pershy1/results/isp/vascular2024_control_chip_endothelium_perturb_output_stats_12L --perturb_stats_prefix vascular2024_control_chip_endothelium_perturb_output_stats_12L

