#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=50:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N isp-tet2-prog

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
cd /grand/GeomicVar/for_bick_group/
conda activate geneformer_env_B/

pip install accelerate

cd /grand/GeomicVar/pershy1/scripts

python geneformer_isp_finetunedclassifier_yp_012724.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/jva_tet2_hscs_imputed_labels/jva_tet2_hscs_imputed_labels.dataset --filter_field organ --filter_class "BM" --model /grand/GeomicVar/pershy1/classifiers/240526_geneformer_CellClassifier_jva_tet2_hscs_imputed_labels_L2048_B6_LR5e-05_LSlinear_WU500_E10_Oadamw_F0 --state_key label --start_state "TET2" --goal_state "AAVS" --embeddings_output_dir /grand/GeomicVar/pershy1/results/embeddings/jva_tet2_hscs_imputed_labels_embeddings_finetuned --state_embs_prefix jva_tet2_hscs_imputed_labels --perturb_output /grand/GeomicVar/pershy1/results/isp/jva_tet2_hscs_imputed_labels_output_12L --perturb_prefix jva_tet2_hscs_imputed_labels_12L --perturb_stats_output /grand/GeomicVar/pershy1/results/isp/jva_tet2_hscs_imputed_labels_output_stats_12L --perturb_stats_prefix jva_tet2_hscs_imputed_labels_output_stats_12L

