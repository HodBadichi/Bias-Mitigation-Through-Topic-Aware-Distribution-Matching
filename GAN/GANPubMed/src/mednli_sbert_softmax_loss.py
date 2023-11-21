"""

"""
from torch.utils.data import DataLoader
import torch
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from pathlib import Path
import logging
from datetime import datetime
import sys
import os
import gzip
import json
from pytorch_lightning.loggers import WandbLogger
import wandb 
import csv
import pandas as pd 


BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5
GENDER= 'all'

nli_train_dataset_path = '/home/liel-blu/project/mednli_data/mli_train_v1.jsonl'
nli_val_dataset_path = '/home/liel-blu/project/mednli_data/mli_dev_v1.jsonl'
nli_test_dataset_path = '/home/liel-blu/project/mednli_data/mli_test_v1.jsonl'
label2int = {"contradiction": 0, "neutral": 1, "entailment": 2}


def wandb_callback(score, epoch, steps):
    wandb.log({'accuracy': score})


def log_eval_results(model_save_path):
    eval_csv = model_save_path / 'eval/accuracy_evaluation_results.csv'
    eval_df = pd.read_csv(eval_csv)
    eval_table = wandb.Table(dataframe=eval_df)
    eval_table_artifact = wandb.Artifact(
    "eval_artifact", 
    type="dataset"
    )
    eval_table_artifact.add(eval_table, "eval_table")
    eval_table_artifact.add_file(eval_csv)
    wandb.log({"eval": eval_table})
    wandb.log_artifact(eval_table_artifact)

def read_jsonl_file(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def generate_train_samples(nli_train_dataset_path):
    train_samples = []
    data = read_jsonl_file(nli_train_dataset_path)
    with open("/home/liel-blu/project/mednli_data/sentences_nli.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(['sentence1', 'sentence2'])
        for row in data:
            label_id = label2int[row['gold_label']]
            csv_writer.writerow([row['sentence1'], row['sentence2']])
            train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    return train_samples

def generate_dev_evaluator(train_loss):
    val_samples = {'female': [], 'male': [], 'unknown': [], 'all': []}
    female_words = ["she", "her", "female", "Her", "She", "Female", "pregnant", "Pregnant", "pregnancy", "Pregnancy", "woman", "Woman", "vagina", "vaginal"]
    male_words = ["he", "his", "male", "He", "His", "Male", "Prostate", "man", "Man", "prostate", "gentleman", "Gentleman"]
    data = read_jsonl_file(nli_val_dataset_path)
    for row in data:
        unstripped_all_words_in_sentence = row['sentence1'].split()+ row['sentence2'].split()
        all_words_in_sentence = [word.replace("*", "") for word in unstripped_all_words_in_sentence]
        label_id = label2int[row['gold_label']]
        val_samples["all"].append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
        if any(word in female_words for word in all_words_in_sentence):
            val_samples['female'].append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
        elif any(word in male_words for word in all_words_in_sentence):
            val_samples['male'].append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
        else:
            val_samples['unknown'].append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    # return EmbeddingSimilarityEvaluator.from_input_examples(val_samples, batch_size=BATCH_SIZE, name='nli-dev')
    val_dataloader = DataLoader(val_samples[GENDER], shuffle=False, batch_size=BATCH_SIZE)
    evaluator = LabelAccuracyEvaluator(val_dataloader, softmax_model = train_loss)
    return evaluator


def train_sbert(model, model_save_path):
    train_samples = generate_train_samples(nli_train_dataset_path)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))
    warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * 0.1) #10% of train data for warm-up
    dev_evaluator = generate_dev_evaluator(train_loss)
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=EPOCHS,
            evaluation_steps=1000,
            optimizer_params={'lr': LEARNING_RATE} ,
            warmup_steps=warmup_steps,
            output_path=str(model_save_path),
            callback=wandb_callback,
            )
    log_eval_results(model_save_path)
    test_sbert(str(model_save_path), train_loss)

def test_sbert(model_save_path,train_loss):
    test_samples = []
    data = read_jsonl_file(nli_test_dataset_path)
    for row in data:
        label_id = label2int[row['gold_label']]
        test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    model = SentenceTransformer(model_save_path).to("cuda")
    test_dataloader = DataLoader(test_samples, shuffle=False, batch_size=BATCH_SIZE)
    tets_evaluator = LabelAccuracyEvaluator(test_dataloader,name="test", softmax_model = train_loss)
    model.evaluate(tets_evaluator)

if __name__ == "__main__":
    # model_name = 'sentence-transformers/all-mpnet-base-v2'
    # model_name = 'all-MiniLM-L6-v2'
    # model_name = "microsoft/biogpt"
    model_name = 'gpt2-medium'
    # model_name = 'bert-base-uncased'
    wandb.init(project="mednli_sbert_softmax_loss", name=f"{GENDER}_{model_name}_epochs={EPOCHS}_lr={LEARNING_RATE}")
    model_save_path = Path('/home/liel-blu/project/saved_models/nli_sbert/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    word_embedding_model = models.Transformer(model_name)
    tokenizer = word_embedding_model.tokenizer
    #add padding token because gpt2 doesn't have a pad token in its tokenizer?

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to('cuda')
    # model = SentenceTransformer(model_name).to('cuda')
    train_sbert(model, model_save_path)
    wandb.finish()
