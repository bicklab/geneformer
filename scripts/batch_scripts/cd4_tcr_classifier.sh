#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=5:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N cd4-tcr

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
cd /grand/GeomicVar/for_bick_group/
conda activate geneformer_env_B/

cd /grand/GeomicVar/pershy1/scripts

python geneformer_build_classifier_yp_012724.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/perturbformer_cd4_tcr_noszabo/perturbformer_cd4_tcr_noszabo.dataset --classification cd4_tcr_noszabo --pretrained_model /grand/GeomicVar/pershy1/Geneformer/geneformer-12L-30M --finetuned_model_output_path /grand/GeomicVar/pershy1/classifiers --label_name IntegratedStim

