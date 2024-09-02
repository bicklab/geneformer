import os
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification, BertConfig
from transformers import Trainer
from transformers.training_args import TrainingArguments
import torch
from geneformer import DataCollatorForCellClassification
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pdb import set_trace
from bertviz import model_view

# Configuration
eval_idx = 100  # You may adjust this if needed

# Load and preprocess dataset
dataset_path = "/grand/GeomicVar/pershy1/data/datasets/jva_runx1_progenitors_imputedlabels_data/jva_runx1_progenitors_imputedlabels.dataset"
train_dataset = load_from_disk(dataset_path)
trainset_organ = train_dataset

organ_list = ["BM"]
celltype_counter = Counter(trainset_organ["label"])

trainset_organ_shuffled = trainset_organ.shuffle(seed=42)
#trainset_organ_shuffled = trainset_organ_shuffled.rename_column("label", "label")

target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
target_name_id_dict = dict(zip(target_names, range(len(target_names))))

def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example

labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
labeled_train_split = labeled_trainset.select(range(round(len(labeled_trainset) * 0.8)))
labeled_eval_split = labeled_trainset.select(range(round(len(labeled_trainset) * 0.8), len(labeled_trainset)))

trained_labels = list(Counter(labeled_train_split["label"]).keys())

def if_trained_label(example):
    return example["label"] in trained_labels

labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

# Prepare datasets
trainset_dict = dict(zip(organ_list, [labeled_train_split]))
evalset_dict = dict(zip(organ_list, [labeled_eval_split_subset]))
traintargetdict_dict = dict(zip(organ_list, [target_name_id_dict]))

# Create directory for heatmaps
output_dir = "/grand/GeomicVar/pershy1/results/jva_runx1_hscs/attention_heatmaps_last_B/"
os.makedirs(output_dir, exist_ok=True)

def save_attention_heatmap(attention_scores, head_index, eval_idx):
    attention_scores = attention_scores.squeeze(0)
    head_scores = attention_scores[head_index]
    plt.figure(figsize=(10, 10))
    sns.heatmap(head_scores, cbar=True, norm=LogNorm(vmin=1e-3, vmax=1e-1))
    plt.title(f'Attention Head {head_index} for eval idx {eval_idx}')
    filename = os.path.join(output_dir, f'attention_head_{head_index}_{eval_idx}.png')
    plt.savefig(filename)
    plt.close()

for organ in organ_list:
    print(organ)
    organ_trainset = trainset_dict[organ]
    organ_evalset = evalset_dict[organ]
    organ_label_dict = traintargetdict_dict[organ]
    input = organ_evalset[eval_idx]['input_ids']
    input_tensor = torch.tensor(input).view(1, -1)

    model_path = "/grand/GeomicVar/pershy1/classifiers/240523_geneformer_CellClassifier_jva_runx1_progenitors_imputedlabels_L2048_B6_LR5e-05_LSlinear_WU500_E10_Oadamw_F0"
    model = BertForSequenceClassification.from_pretrained(model_path,
                                                          num_labels=len(organ_label_dict), 
                                                          output_attentions=True, 
                                                          output_hidden_states=False)
    
    model.eval()
    input_tensor = input_tensor.to(model.device)
    
    with torch.no_grad():
        predictions = model(input_tensor)
    
    attention = predictions[-1]
    attention_last = attention[-1]
    
    for head_index in range(attention_last.shape[1]):
        save_attention_heatmap(attention_last, head_index, eval_idx)
