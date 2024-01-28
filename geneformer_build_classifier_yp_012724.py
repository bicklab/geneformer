# GeneFormer Classification Fine-Tuning Script
# Author: Yash Pershad
# Date: 01/27/2024
#
# This script fine-tunes a pre-trained generalized model into a classifier.
# It uses command-line arguments for specifying parameters such as dataset path, model configuration, and training settings.

# Example run: 
# python geneformer_build_classifier_yp_012724.py \
# --path_to_train_dataset “/grand/GeomicVar/pershy1/scRNAseq_control_pbmcs_labeled/scRNAseq_control_pbmcs_labeled.dataset” \
# --classification “pbmc_ctrlvsil6” \
# --pretrained_model “/grand/GeomicVar/pershy1/Geneformer/“ \
# --finetuned_model_output_path "/grand/GeomicVar/pershy1/classifiers" \
# --label_name “group”

# Note: This script does not filter by cell-type. Might could consider filtering by cell type before feeding in dataset


import argparse
import os
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    # Function to parse command line arguments
    parser = argparse.ArgumentParser(description='GeneFormer Classification Script')

    # Adding arguments with default values where specified
    parser.add_argument('--path_to_train_dataset', required=True)
    parser.add_argument('--classification', required=True)
    parser.add_argument('--pretrained_model', default='/grand/GeomicVar/pershy1/Geneformer/')
    parser.add_argument('--finetuned_model_output_path', required=True)
    parser.add_argument('--label_name', required=True)
    parser.add_argument('--train_test_split', type=float, default=0.8)
    parser.add_argument('--max_input_size', type=int, default=2 ** 11)
    parser.add_argument('--max_lr', type=float, default=5e-5)
    parser.add_argument('--freeze_layers', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_proc', type=int, default=16)
    parser.add_argument('--geneformer_batch_size', type=int, default=12)
    parser.add_argument('--lr_schedule_fn', default='linear')
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optimizer', default='adamw')

    return parser.parse_args()

def compute_metrics(pred):
    # Function to compute evaluation metrics
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }


def get_training_data(path_to_train_dataset, label_name, train_test_split):
    # Function to prepare training and evaluation datasets
    dataset_list = []
    evalset_list = []
    organ_list = []
    target_dict_list = []

    train_dataset=load_from_disk(path_to_train_dataset)
    trainset_shuffled = train_dataset.shuffle(seed=42)
    trainset_shuffled_labeled = trainset_shuffled.rename_column(label_name, "label")
    target_names = list(Counter(trainset_shuffled_labeled["label"]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    target_dict_list += [target_name_id_dict]

    def classes_to_ids(example):
        # Function to convert class labels to numerical IDs
        example["label"] = target_name_id_dict[example["label"]]
        return example

    labeled_trainset = trainset_shuffled_labeled.map(classes_to_ids, num_proc=16)
    # create 80/20 train/eval splits
    labeled_train_split = labeled_trainset.select([i for i in range(0,round(len(labeled_trainset)*train_test_split))])
    labeled_eval_split = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*train_test_split),len(labeled_trainset))])

    # filter dataset for cell types in corresponding training set
    trained_labels = list(Counter(labeled_train_split["label"]).keys())

    def if_trained_label(example):
        # Function to filter dataset based on trained labels
        return example["label"] in trained_labels

    labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)
    
    dataset_list += [labeled_train_split]
    evalset_list += [labeled_eval_split_subset]

    organ_trainset = dataset_list[0]
    organ_evalset = evalset_list[0]
    organ_label_dict = target_dict_list[0]

    return organ_trainset, organ_evalset, organ_label_dict

def main():
    # Main function to run the script
    args = parse_arguments()

    # Use the arguments in the script
    path_to_train_dataset = args.path_to_train_dataset
    classification = args.classification
    pretrained_model = args.pretrained_model
    finetuned_model_output_path = args.finetuned_model_output_path
    label_name = args.label_name
    train_test_split = args.train_test_split
    max_input_size = args.max_input_size
    max_lr = args.max_lr
    freeze_layers = args.freeze_layers
    num_gpus = args.num_gpus
    num_proc = args.num_proc
    geneformer_batch_size = args.geneformer_batch_size
    lr_schedule_fn = args.lr_schedule_fn
    warmup_steps = args.warmup_steps
    epochs = args.epochs
    optimizer = args.optimizer

    organ_trainset, organ_evalset, organ_label_dict = get_training_data(path_to_train_dataset, label_name, train_test_split)

    # set logging steps
    logging_steps = round(len(organ_trainset)/geneformer_batch_size/10)

    # reload pretrained model
    model = BertForSequenceClassification.from_pretrained(pretrained_model, 
                                                      num_labels=len(organ_label_dict.keys()),
                                                      output_attentions = False,
                                                      output_hidden_states = False).to("cuda")

    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_dir = f"{finetuned_model_output_path}/{datestamp}_geneformer_CellClassifier_{classification}_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"

    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)
        
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }
    
    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=organ_trainset,
        eval_dataset=organ_evalset,
        compute_metrics=compute_metrics
    )
    # train the cell type classifier
    trainer.train()
    predictions = trainer.predict(organ_evalset)
    with open(f"{output_dir}predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()