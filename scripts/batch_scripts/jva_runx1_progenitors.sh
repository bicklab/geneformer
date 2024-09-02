#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=50:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N isp-runx1-prog

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
cd /grand/GeomicVar/for_bick_group/
conda activate geneformer_env_B/

pip install accelerate

cd /grand/GeomicVar/pershy1/scripts

python geneformer_isp_finetunedclassifier_yp_012724.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/jva_runx1_progenitors_imputedlabels_data/jva_runx1_progenitors_imputedlabels.dataset --filter_field ident --filter_class 1 --model /grand/GeomicVar/pershy1/classifiers/240523_geneformer_CellClassifier_jva_runx1_progenitors_imputedlabels_L2048_B6_LR5e-05_LSlinear_WU500_E10_Oadamw_F0 --state_key label --start_state "RUNX1"  --goal_state "Control" --embeddings_output_dir /grand/GeomicVar/pershy1/results/embeddings/jva_runx1_progenitors_imputedlabels_embeddings_finetuned --state_embs_prefix jva_runx1_progenitors_imputedlabels --perturb_output /grand/GeomicVar/pershy1/results/isp/jva_runx1_progenitors_imputedlabels_output_12L --perturb_prefix jva_runx1_progenitors_imputedlabels_12L --perturb_stats_output /grand/GeomicVar/pershy1/results/isp/jva_runx1_progenitors_imputedlabels_output_stats_12L --perturb_stats_prefix jva_runx1_progenitors_imputedlabels_output_stats_12L

