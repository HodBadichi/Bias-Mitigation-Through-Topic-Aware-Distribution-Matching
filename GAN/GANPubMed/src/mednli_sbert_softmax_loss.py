"""

"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import json
from pytorch_lightning.loggers import WandbLogger
import wandb 


nli_train_dataset_path = '/home/liel-blu/project/mednli_data/mli_train_v1.jsonl'
nli_dev_dataset_path = '/home/liel-blu/project/mednli_data/mli_dev_v1.jsonl'
nli_test_dataset_path = '/home/liel-blu/project/mednli_data/mli_test_v1.jsonl'
label2int = {"contradiction": 0, "neutral": 1, "entailment": 2}
train_batch_size = 16
num_epochs = 1

def wandb_callback(score, loss_value, epoch, steps):
    wandb.log({'epoch': epoch, 'max_average_precision': score, 'loss': loss_value})


def log_eval_results(model_save_path):
    eval_csv = model_save_path / 'eval/binary_classification_evaluation_results.csv'
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
    for row in data:
        label_id = label2int[row['gold_label']]
        train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    return train_samples


def generate_dev_evaluator(nli_val_dataset_path):
    val_samples = []
    data = read_jsonl_file(nli_val_dataset_path)
    for row in data:
        label_id = label2int[row['gold_label']]
        val_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    # return EmbeddingSimilarityEvaluator.from_input_examples(val_samples, batch_size=train_batch_size, name='nli-dev')
    val_dataloader = DataLoader(val_samples, shuffle=False, batch_size=train_batch_size)
    evaluator = LabelAccuracyEvaluator(val_dataloader,model_save_path)
    return evaluator


def train_sbert(model, model_save_path):
    train_samples = generate_train_samples(nli_train_dataset_path)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    dev_evaluator = generate_dev_evaluator()
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            callback=wandb_callback
            )
    log_eval_results(model_save_path)

def test_sbert(model_save_path):
    test_samples = []
    data = read_jsonl_file(nli_test_dataset_path)
    for row in data:
        label_id = label2int[row['gold_label']] / 2 # Normalize score to range 0 ... 1
        test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    model = SentenceTransformer(model_save_path)
    test_dataloader = DataLoader(val_samples, shuffle=False, batch_size=train_batch_size)
    tets_evaluator = LabelAccuracyEvaluator(test_dataloader,model_save_path)
    model.evaluate(tets_evaluator)

if __name__ == "__main__":
    wandb.init(project="mednli_sbert_softmax_loss")
    model_name = 'bert-base-uncased'
    model_save_path = '/home/liel-blu/project/saved_models/nli_sbert/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_sbert(model, model_save_path)
    test_sbert(model_save_path)
    wandb.finish()
