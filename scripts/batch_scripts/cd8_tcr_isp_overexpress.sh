#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=20:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N cd8-tcr

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
cd /grand/GeomicVar/for_bick_group/
conda activate geneformer_env_B/

cd /grand/GeomicVar/pershy1/scripts

python geneformer_isp_finetunedclassifier_yp_012724.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/perturbformer_cd8_tcr_noszabo/perturbformer_cd8_tcr_noszabo.dataset --filter_field IntegratedCellType --filter_class "CD8" --model /grand/GeomicVar/pershy1/classifiers/240623_geneformer_CellClassifier_cd8_tcr_noszabo_L2048_B6_LR5e-05_LSlinear_WU500_E10_Oadamw_F0 --state_key IntegratedStim --start_state "cd3cd28"  --goal_state "unstim" --perturb_type "overexpress" --embeddings_output_dir /grand/GeomicVar/pershy1/results/embeddings/perturbformer_cd8_tcr_noszabo_finetuned --state_embs_prefix perturbformer_cd8_tcr_noszabo --perturb_output /grand/GeomicVar/pershy1/results/isp/perturbformer_cd8_tcr_noszabo_perturb_output_overexpress_12L --perturb_prefix perturbformer_cd8_tcr_noszabo_perturb_output_overexpress_12L --perturb_stats_output /grand/GeomicVar/pershy1/results/isp/perturbformer_cd8_tcr_noszabo_perturb_output_stats_overexpress_12L --perturb_stats_prefix perturbformer_cd8_tcr_noszabo_perturb_output_stats_overexpress_12L

