#!/bin/bash -l
#PBS -A GeomicVar
#PBS -l walltime=3:00:00
#PBS -l filesystems=grand
#PBS -l select=1:ngpus=4:gputype=A100:system=polaris
#PBS -q preemptable
##PBS -q debug-scaling
##PBS -q debug
#PBS -N enl6it-classifier

cd ${PBS_O_WORKDIR}
echo ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
cd /grand/GeomicVar/pershy1/conda/
conda activate new_ssl_test/

cd /grand/GeomicVar/pershy1/scripts

python geneformer_prepare_dataset_yp_09292024_lippmann.py --path_to_loom /grand/GeomicVar/pershy1/data/looms/enl6it_ensembl.loom --path_to_reformatted_loom /grand/GeomicVar/pershy1/data/reformatted_looms/lippmann_enl6it/enl6it_ensembl.loom --path_to_loom_directory /grand/GeomicVar/pershy1/data/reformatted_looms/lippmann_enl6it/ --output_dir /grand/GeomicVar/pershy1/data/datasets/lippmann_enl6it --output_prefix lippmann_enl6it

python geneformer_build_classifier_yp_012724.py --path_to_train_dataset /grand/GeomicVar/pershy1/data/datasets/lippmann_enl6it/lippmann_enl6it.dataset --classification enl6it --pretrained_model /grand/GeomicVar/pershy1/Geneformer/geneformer-12L-30M --finetuned_model_output_path /grand/GeomicVar/pershy1/classifiers --label_name type