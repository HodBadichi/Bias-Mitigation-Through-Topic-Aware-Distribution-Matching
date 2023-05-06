import pandas as pd
import torch
from itertools import combinations
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import pytz


def create_labeld_dataset_from_abstract(sentences_list):
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
    dataset_df_0 = dataset_df[dataset_df['label'] == 0]
    dataset_df_1 = dataset_df[dataset_df['label'] == 1]
    dataset_df_0 = dataset_df_0.sample(
        n=dataset_df_1.shape[0], random_state=42)
    dataset_df = pd.concat([dataset_df_0, dataset_df_1])
    dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(
        drop=True)  # shuffle the rows
    return dataset_df


def train_dataset_to_input_examples(train_dataset):
    train_examples = []
    for _, row in train_dataset.iterrows():
        train_examples.append(InputExample(
            texts=[row['sentence_1'], row['sentence_2']], label=row['label']))
    return train_examples


def test_dataset_to_evaluator(test_dataset):
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator(
        test_dataset['sentence_1'], test_dataset['sentence_2'], test_dataset['label'])
    return test_evaluator


def train_sentence_transformer(train_dataset, test_dataset):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_examples = train_dataset_to_input_examples(train_dataset)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.OnlineContrastiveLoss(
        model=model)  # can't use BCEWithLogitsLoss here
    model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=test_dataset_to_evaluator(
        test_dataset), output_path='best_model', save_best_model=True, epochs=10, warmup_steps=100, show_progress_bar=True)


def prepare_data():
    documents_df = pd.read_csv(
        f'../../../data/abstract_2005_2020_gender_and_topic.csv', encoding='utf8')
    abstracts_df = documents_df[['PMID', 'title_and_abstract']]
    d = {}
    for _, row in abstracts_df.iterrows():
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
    # logger = WandbLogger(name="test",
    # version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
    # project='SentenceTransformerTraining')
    train_data, test_data = prepare_data()
    train_sentence_transformer(train_data, test_data)


if __name__ == '__main__':
    run()
