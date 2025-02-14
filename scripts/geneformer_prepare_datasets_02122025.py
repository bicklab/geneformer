# GeneFormer Prepare Drug Datasets
# Author: Yash Pershad
# Date: 11/5/2024
#
# This script takes a loom file with specific attributes and converts it into a tokenized .dataset file 
# that is readable by PyTorch. The script is intended for use with datasets containing 'group', 
# 'n_counts', and 'Gene' (Ensembl ID) attributes.

# Dependencies: anndata, loompy, datasets, transformers, tdigest, scanpy

# Example to run:
# python geneformer_prepare_dataset_yp_012724.py \
# --path_to_loom "control_pbmc_ensembl.loom" \ 
# --path_to_reformatted_loom "control_pbmc_ensembl_reformatted.loom" \
# --path_to_loom_directory "/home/jupyter/bicklab-pershad/edit/reformatted_looms" \
# --output_dir "scRNAseq_control_pbmcs" \
# --output_prefix "scRNAseq_control_pbmcs"

# Assumes the organ of all data is "blood"

import argparse
from geneformer import TranscriptomeTokenizer
import loompy

def parse_arguments():
    # Parses command line arguments
    parser = argparse.ArgumentParser(description='GeneFormer Tokenizer Script')

    # Adding arguments with default values where specified
    parser.add_argument('--path_to_loom_directory', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_prefix', required=True)
    return parser.parse_args()

def main():
    # Main function to run the script
    args = parse_arguments()

    # Extracting arguments for use in the script
    path_to_loom_directory = args.path_to_loom_directory
    output_dir = args.output_dir
    output_prefix = args.output_prefix

    # Make reformatted loom into tokenized dataset
    tk = TranscriptomeTokenizer({"cell_type": "cell_type", 
                                 "n_counts": "nCount_RNA", 
                                 "Label": "Label"}, nproc=16)
    print(path_to_loom_directory)
    tk.tokenize_data(str(path_to_loom_directory), 
                    str(output_dir), 
                    str(output_prefix), 
                    file_format="loom")

if __name__ == "__main__":
    main()
