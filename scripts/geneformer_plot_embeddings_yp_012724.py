# GeneFormer Plot Embeddings of Pretrained and Fine-Tuned Models
# Author: Yash Pershad
# Date: 01/27/2024
#
# This script plots embeddings of your cells of interest for 1) a pretrained model and a 2) fine-tuned model.

# Example to run:
# python geneformer_plot_embeddings_yp_012724.py \
# --path_to_train_dataset “/grand/GeomicVar/pershy1/scRNAseq_control_pbmcs_labeled/scRNAseq_control_pbmcs_labeled.dataset” \
# --filter_field “cell_type” \
# --filter_class "CD14 Monos" \
# --pretrained_model “/grand/GeomicVar/pershy1/Geneformer/“ \
# --finetuned_model “/grand/GeomicVar/pershy1/classifiers/" \
# --label_to_plot “group” \
# --embeddings_output_dir “/grand/GeomicVar/pershy1/control_pbmc_embeddings” \
# --state_embs_prefix “control_pbmc” \
# --plots_output “/grand/GeomicVar/pershy1/control_pbmc_plots” \
# —-plot_output_prefix “control_pbmc” 

from geneformer import EmbExtractor

import argparse
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    # Function to parse command line arguments
    parser = argparse.ArgumentParser(description='GeneFormer Plot Embeddings Script')

    # Adding arguments with default values where specified
    parser.add_argument('--path_to_train_dataset', required=True)
    parser.add_argument('--filter_field', required=True)
    parser.add_argument('--filter_class', type=str, required=True)
    parser.add_argument('--pretrained_model', default='/grand/GeomicVar/pershy1/Geneformer/')
    parser.add_argument('--finetuned_model', required=True)
    parser.add_argument('--label_to_plot', required=True)
    parser.add_argument('--embeddings_output_dir', required=True)
    parser.add_argument('--state_embs_prefix', required=True)
    parser.add_argument('--plots_output_dir', required=True)
    parser.add_argument('--plot_output_prefix', required=True)
    return parser.parse_args()

def extract_embeddings(filter_data_dict, model, label_to_plot, input_dataset, embeddings_output_dir, state_embs_prefix):
    # initiate EmbExtractor
    embex = EmbExtractor(model_type="CellClassifier",
                        num_classes=2,
                        filter_data=filter_data_dict,
                        max_ncells=1000,
                        #max_ncells=10000,
                        emb_layer=0,
                        emb_label=[label_to_plot],
                        labels_to_plot=[label_to_plot],
                        forward_batch_size=100,
                        nproc=16)
        
    # extracts embedding from input data
    embs = embex.extract_embs(model,
                            input_dataset,
                            embeddings_output_dir,
                            state_embs_prefix)
    
    return embex, embs

def plot_embeddings(embex, embs, plot_output_dir, plot_output_prefix):
    # plot UMAP of cell embeddings
    embex.plot_embs(embs=embs, 
                plot_style="umap",
                output_directory=plot_output_dir,  
                output_prefix=plot_output_prefix)

    # plot heatmap of cell embeddings
    embex.plot_embs(embs=embs, 
                    plot_style="heatmap",
                    output_directory=plot_output_dir,
                    output_prefix=plot_output_prefix)


def main():
    # Main function to run the script
    args = parse_arguments()

    # Use the arguments in the script
    path_to_train_dataset = args.path_to_train_dataset
    filter_field = args.filter_field
    filter_class = args.filter_class
    pretrained_model = args.pretrained_model
    finetuned_model = args.finetuned_model
    label_to_plot = args.label_to_plot
    embeddings_output_dir = args.embeddings_output_dir
    state_embs_prefix = args.state_embs_prefix
    plots_output_dir = args.plots_output_dir
    plot_output_prefix = args.plot_output_prefix

    #setting up filter for class of interest (e.g., cell type)
    filter_data_dict = {filter_field: [filter_class]}

    #extract and plot embeddings for pretrained model
    pretrained_suffix = "_pretrained"
    pretrained_embex, pretrained_embs = extract_embeddings(filter_data_dict, pretrained_model, label_to_plot, 
                       path_to_train_dataset, embeddings_output_dir + pretrained_suffix, 
                       state_embs_prefix + pretrained_suffix)
    plot_embeddings(pretrained_embex, pretrained_embs, plots_output_dir + pretrained_suffix, plot_output_prefix + pretrained_suffix) 

    #extract and plot embeddings for fine-tuned model
    finetuned_suffix = "_finetuned"
    finetuned_embex, finetuned_embs = extract_embeddings(filter_data_dict, finetuned_model, label_to_plot, 
                       path_to_train_dataset, embeddings_output_dir + finetuned_suffix, 
                       state_embs_prefix + finetuned_suffix)
    plot_embeddings(finetuned_embex, finetuned_embs, plots_output_dir + finetuned_suffix, plot_output_prefix + finetuned_suffix)

if __name__ == "__main__":
    main()
