import os
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from datetime import datetime

"""
This module trains a sentence transformer model on the Pubmed abstracts dataset with binary labels.
The label is 1 when two sentences are adjacent in the abstract and 0 otherwise.
0 label is given only to sentences from the same abstract.
We used the following as guides to write this module:
1. https://www.sbert.net/docs/training/overview.html
2. https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/quora_duplicate_questions/training_OnlineContrastiveLoss.py
"""


def create_labeld_dataset_from_abstract(sentences_list):
    """assigns the label 1 to adjacent sentences and 0 to non-adjacent sentences

    Args:
        sentences_list (list): a list of all the sentences in the abstract

    Returns:
        dict: a dictionary with pair of sentences as keys and binary label as value
    """
    dataset_dict = {}
    adjacent_sentences_pairs = list(
        zip(sentences_list[:-1], sentences_list[1:]))
    all_pairs_of_sentences = combinations(sentences_list, 2)
    for pair in all_pairs_of_sentences:
        if pair in adjacent_sentences_pairs:
            dataset_dict[pair] = 1
        else:
            dataset_dict[pair] = 0
    return dataset_dict


def balance_dataset(dataset_df):
    """Balances the dataset by having the same number of 0 and 1 labels
    using 

    Args:
        dataset_df (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label

    Returns:
        Pandas DataFrame: a balanced dataframe
    """
    dataset_df_0 = dataset_df[dataset_df['label'] == 0]
    dataset_df_1 = dataset_df[dataset_df['label'] == 1]
    dataset_df_0 = dataset_df_0.sample(
        n=dataset_df_1.shape[0], random_state=42)
    dataset_df = pd.concat([dataset_df_0, dataset_df_1])
    dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(
        drop=True)  # shuffle the rows
    return dataset_df


def train_dataset_to_input_examples(train_dataset):
    """Creates a list of InputExamples from the train dataset

    Args:
        train_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label

    Returns:
       list: a list of InputExamples
    """
    train_examples = []
    for _, row in train_dataset.iterrows():
        train_examples.append(InputExample(
            texts=[row['sentence_1'], row['sentence_2']], label=row['label']))
    return train_examples


def test_dataset_to_evaluator(test_dataset):
    """Creates an evaluator for the test dataset to be used in the fit method of the sentence transformer model
    The evaluator is for binary labels:
      https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/BinaryClassificationEvaluator.py

    Args:
        test_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label
    Returns:
        BinaryClassificationEvaluator: _description_
    """
    test_evaluator = evaluation.BinaryClassificationEvaluator(
        test_dataset['sentence_1'].to_list(), test_dataset['sentence_2'].to_list(), test_dataset['label'].to_list())
    return test_evaluator


def train_sentence_transformer(train_dataset, test_dataset):
    """Train a sentence transformer model using the OnlineContrastiveLoss with 
    distance metric and margin as used in sbert tutorial for duplicate questions (binary labels)

    Args:
        train_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label
        test_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_examples = train_dataset_to_input_examples(train_dataset)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    margin = 0.5
    train_loss = losses.OnlineContrastiveLoss(
        model=model, distance_metric=distance_metric, margin=margin)
    model_save_path = 'best_model/output/training_OnlineConstrativeLoss10Epochs-' + \
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(model_save_path, exist_ok=True)
    model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=test_dataset_to_evaluator(
        test_dataset), output_path=model_save_path, save_best_model=True, epochs=10, warmup_steps=100, show_progress_bar=True)


def prepare_data():
    """Uses the abstracts dataset to create a dataset with sentence pairs and labels.

    Returns:
        Pandas DataFrame: a split to train and test datasets.
    """
    documents_df = pd.read_csv(
        f'../../../data/abstract_2005_2020_gender_and_topic.csv', encoding='utf8')
    abstracts_df = documents_df[['PMID', 'title_and_abstract']]
    d = {}
    for i, row in abstracts_df.iterrows():
        sentences_list = (row['title_and_abstract'].split(';'))
        d.update(create_labeld_dataset_from_abstract(sentences_list))
    dataset_dict = {'sentence_1': [k[0] for k in d.keys()], 'sentence_2': [
        k[1] for k in d.keys()], 'label': list(d.values())}
    dataset_df = pd.DataFrame(dataset_dict)
    dataset_df = balance_dataset(dataset_df)
    train_data, test_data = train_test_split(
        dataset_df, test_size=0.2, random_state=42)
    return train_data, test_data


def run():
    train_data, test_data = prepare_data()
    train_sentence_transformer(train_data, test_data)


if __name__ == '__main__':
    run()
