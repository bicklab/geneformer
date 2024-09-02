#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=20:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N cd8-lps

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
cd /grand/GeomicVar/for_bick_group/
conda activate geneformer_env_B/

pip install accelerate

cd /grand/GeomicVar/pershy1/scripts

python geneformer_isp_finetunedclassifier_buenrostro_05312024.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/buenrostro_stim_cd8_lps6hr/buenrostro_stim_cd8_lps6hr.dataset --model /grand/GeomicVar/pershy1/classifiers/240606_geneformer_CellClassifier_buenrostro_stim_cd8_lps6hr_L2048_B6_LR5e-05_LSlinear_WU500_E10_Oadamw_F0 --state_key Condition --start_state "LPS_6h"  --goal_state "Control_6h" --embeddings_output_dir /grand/GeomicVar/pershy1/results/embeddings/buenrostro_stim_cd8_lps6hr_classifier_embeddings_finetuned --state_embs_prefix buenrostro_stim_cd8_lps6hr --perturb_output /grand/GeomicVar/pershy1/results/isp/buenrostro_stim_cd8_lps6hr_output_12L --perturb_prefix buenrostro_stim_cd8_lps6hr_12L --perturb_stats_output /grand/GeomicVar/pershy1/results/isp/buenrostro_stim_cd8_lps6hr_output_stats_12L --perturb_stats_prefix buenrostro_stim_cd8_lps6hr_output_stats_12L

