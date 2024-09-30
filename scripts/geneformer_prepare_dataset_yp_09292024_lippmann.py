# GeneFormer Convert Loompy to Tokenized Dataset
# Author: Yash Pershad
# Date: 01/27/2024
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
import pandas as pd
import numpy as np

def parse_arguments():
    # Parses command line arguments
    parser = argparse.ArgumentParser(description='GeneFormer Tokenizer Script')

    # Adding arguments with default values where specified
    parser.add_argument('--path_to_loom', required=True)
    parser.add_argument('--path_to_reformatted_loom', required=True)
    parser.add_argument('--path_to_loom_directory', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_prefix', required=True)
    return parser.parse_args()

def reformat_loom(ds, filename):
    # Function to reformat the loom file
    # This includes renaming and reorganizing attributes to the required format
    
    col_attrs = {
            "n_counts": np.array(ds.ca['nCount_RNA']),
            "type": np.array(ds.ca['type'])
    }

    # Reformat row attributes
    row_attrs = {'ensembl_id': list(dict(ds.ra.items())['Gene'])}

    return loompy.create(filename, ds[:, :], row_attrs, col_attrs)

def main():
    # Main function to run the script
    args = parse_arguments()

    # Extracting arguments for use in the script
    path_to_loom = args.path_to_loom
    path_to_reformatted_loom = args.path_to_reformatted_loom
    path_to_loom_directory = args.path_to_loom_directory
    output_dir = args.output_dir
    output_prefix = args.output_prefix

    # Read in loom object
    ds = loompy.connect(path_to_loom)

    # Reformat loom object
    reformat_loom(ds, path_to_reformatted_loom)

    ds.close() # Close loom object after use

    # Make reformatted loom into tokenized dataset
    tk = TranscriptomeTokenizer({"n_counts": "nCount_RNA",
                                 "type": "type"}, nproc=16)
    print(path_to_loom_directory)
    tk.tokenize_data(str(path_to_loom_directory), 
                    str(output_dir), 
                    str(output_prefix), 
                    file_format="loom")

if __name__ == "__main__":
    main()
