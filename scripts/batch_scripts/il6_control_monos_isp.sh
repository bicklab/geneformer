#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=40:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N control-il6

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module load conda
cd /grand/GeomicVar/for_bick_group/
conda activate geneformer_env_B/

cd /grand/GeomicVar/pershy1/scripts

python geneformer_isp_finetunedclassifier_yp_012724.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/hpb2022_il6_control_monos/hpb2022_il6_control_monos.dataset --filter_field cell_type --filter_class "CD14 Monos" --model /grand/GeomicVar/pershy1/classifiers/240418_geneformer_CellClassifier_control_monos_nostim_il6_L2048_B6_LR5e-05_LSlinear_WU500_E10_Oadamw_F0 --state_key group --start_state "none VEH"  --goal_state "none STIM" --embeddings_output_dir /grand/GeomicVar/pershy1/results/embeddings/hpb2022_il6_control_monos_embeddings_finetuned --state_embs_prefix hpb2022_il6_control_monos --perturb_output /grand/GeomicVar/pershy1/results/isp/hpb2022_il6_control_monos_perturb_output_12L --perturb_prefix hpb2022_il6_control_monos_perturb_output_12L --perturb_stats_output /grand/GeomicVar/pershy1/results/isp/hpb2022_il6_control_monos_perturb_output_stats_12L --perturb_stats_prefix hpb2022_il6_control_monos_perturb_output_stats_12L

