# GeneFormer In Silico Perturbation with Fine-Tuned Model
# Author: Yash Pershad
# Date: 01/27/2024
#
# This script performs in silico perturbation for a dataset given a fine-tuned model. 
# It is designed to work with datasets having 2 cell states and no alternative states.
# The script calculates the state embeddings and performs perturbation analysis on the dataset.

# Example to run:
# python geneformer_isp_finetunedclassifier_yp_012724.py \
# --path_to_train_dataset “/grand/GeomicVar/pershy1/scRNAseq_control_pbmcs_labeled/scRNAseq_control_pbmcs_labeled.dataset” \
# --filter_field “cell_type” \
# --filter_class “CD14 Monos” \
# --model “/grand/GeomicVar/pershy1/classifiers/" \
# --state_key “group”  \
# --start_state “none VEH” \
# --goal_state “non STIM ” \
# --embeddings_output_dir “/grand/GeomicVar/pershy1/control_pbmc_embeddings_finetuned" \
# --state_embs_prefix “control_pbmc_finetuned” \
# --perturb_output “/grand/GeomicVar/pershy1/all_pbmc_il6stim_perturb_output” \
# --perturb_prefix “all_pbmc_il6stim_perturb_output” 
# --perturb_stats_output "/grand/GeomicVar/pershy1/all_pbmc_il6stim_perturb_output_stats"
# --perturb_stats_prefix "all_pbmc_il6stim_perturb_output_stats"

from geneformer import InSilicoPerturber, InSilicoPerturberStats, EmbExtractor

import argparse
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    # Function to parse command line arguments
    parser = argparse.ArgumentParser(description='GeneFormer In Silico Perturbation Script')

    # Adding arguments with default values where specified
    parser.add_argument('--path_to_train_dataset', required=True)
    parser.add_argument('--filter_field', required=True)
    parser.add_argument('--filter_class', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--state_key', required=True)
    parser.add_argument('--start_state', required=True)
    parser.add_argument('--goal_state', required=True)
    parser.add_argument('--embeddings_output_dir', required=True)
    parser.add_argument('--state_embs_prefix', required=True)
    parser.add_argument('--perturb_type', default='delete')
    parser.add_argument('--gene_list', nargs='+', default='all') #remember must be Ensembl ID 
    # e.g., ["ENSG00000168769", "ENSG00000119772", "ENSG00000171456", "ENSG00000096968", "ENSG00000141510"]
    parser.add_argument('--perturb_output', required=True)
    parser.add_argument('--perturb_prefix', required=True)
    parser.add_argument('--perturb_stats_output', required=True)
    parser.add_argument('--perturb_stats_prefix', required=True)
    return parser.parse_args()

def main():
    # Main function to run the script
    args = parse_arguments()

    # Extracting arguments for use in the script
    path_to_train_dataset = args.path_to_train_dataset
    filter_field = args.filter_field
    filter_class = args.filter_class
    model = args.model
    state_key = args.state_key
    start_state = args.start_state
    goal_state = args.goal_state
    embeddings_output_dir = args.embeddings_output_dir
    state_embs_prefix = args.state_embs_prefix
    perturb_type = args.perturb_type
    gene_list = args.gene_list
    print(gene_list)
    perturb_output = args.perturb_output
    perturb_prefix = args.perturb_prefix
    perturb_stats_output = args.perturb_stats_output
    perturb_stats_prefix = args.perturb_stats_prefix

    # Setting up filter for class of interest (e.g., cell type)
    #filter_data_dict = {filter_field: [filter_class]}

    # Obtain start, goal, and alternative embedding positions
    # Separate from perturb_data to avoid repeating calculations when parallelizing
    embex = EmbExtractor(model_type="CellClassifier",
                        num_classes=2,
                        #filter_data=filter_data_dict,
                        max_ncells=1000,
                        emb_layer=0,
                        summary_stat="exact_mean",
                        forward_batch_size=100,
                        nproc=16)
    
    cell_states_to_model={"state_key": state_key, 
                        "start_state": start_state, 
                        "goal_state": goal_state,
                        "alt_states": []}
    
    state_embs_dict = embex.get_state_embs(cell_states_to_model,
                                       model,
                                       path_to_train_dataset,
                                       embeddings_output_dir,
                                       state_embs_prefix)

    print("---------Embeddings extracted-------\n")

    # Setting up in silico perturbation
    isp = InSilicoPerturber(perturb_type=perturb_type,
                            perturb_rank_shift=None,
                            genes_to_perturb=gene_list,
                            combos=0,
                            anchor_gene=None,
                            model_type="CellClassifier",
                            num_classes=2,
                            emb_mode="cell",
                            cell_emb_style="mean_pool",
                            #filter_data=filter_data_dict,
                            cell_states_to_model=cell_states_to_model,
                            state_embs_dict=state_embs_dict,
                            max_ncells=2000,
                            emb_layer=0,
                            forward_batch_size=100,
                            nproc=16)


    print("-------------------------- Perturbation study beginning ----------- \n")
    # Perform in silico perturbation and output intermediate files
    isp.perturb_data(model,
                    path_to_train_dataset,
                    perturb_output,
                    perturb_prefix)

    # Setting up in silico perturbation statistics
    ispstats = InSilicoPerturberStats(mode="goal_state_shift",
                                    genes_perturbed=gene_list, 
                                    combos=0,
                                    anchor_gene=None,
                                    cell_states_to_model=cell_states_to_model)

    # Extract and process statistics from perturbation output in a .csv
    ispstats.get_stats(perturb_output,
                    None,
                    perturb_stats_output,
                    perturb_stats_prefix)

if __name__ == "__main__":
    main()
