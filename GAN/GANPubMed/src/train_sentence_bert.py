import os
import pandas as pd
from itertools import combinations
from pathlib import Path
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, LoggingHandler
from torch.utils.data import DataLoader
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
import wandb 
import pytz
import torch

"""
This module trains a sentence transformer model on the Pubmed abstracts dataset with binary labels.
The label is 1 when two sentences are adjacent in the abstract and 0 otherwise.
0 label is given only to sentences from the same abstract.
We used the following as guides to write this module:
1. https://www.sbert.net/docs/training/overview.html
2. https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/quora_duplicate_questions/training_OnlineContrastiveLoss.py
"""

EPOCHS = 20
MODEL = 'all-MiniLM-L6-v2'
SAVE_PATH = Path(__file__).resolve().parents[4] / 'saved_models' /'sentence_bert_training' 
SAVE_PATH.mkdir(parents=True, exist_ok=True)



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

def create_adjacent_pairs_from_abstract(sentences_list):
    dataset_dict = {}
    adjacent_sentences_pairs = list(
        zip(sentences_list[:-1], sentences_list[1:]))
    for pair in adjacent_sentences_pairs:
        dataset_dict[pair] = 1
    return dataset_dict

def create_same_abstract_pairs(sentences_list):
    dataset_dict = {}
    all_pairs_of_sentences = combinations(sentences_list, 2)
    for pair in all_pairs_of_sentences:
        dataset_dict[pair] = 1
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
    try:
        if dataset_df_0.shape[0] > dataset_df_1.shape[0]:
            dataset_df_0 = dataset_df_0.sample(
            n=dataset_df_1.shape[0], random_state=42)
    except:
        print("HOW CAN THERE BE MORE 1 LABELS THAN 0 LABELS?")
        print(dataset_df_0, dataset_df_1)
        raise
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


def get_loss(loss_model, train_objectives, model):
    losses = []
    dataloaders = [dataloader for dataloader, _ in train_objectives]
    for dataloader in dataloaders:
        dataloader.collate_fn = model.smart_batching_collate
    num_train_objectives = len(train_objectives)
    steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
    data_iterators = [iter(dataloader) for dataloader in dataloaders]
    for _ in range(steps_per_epoch):  
        for train_idx in range(num_train_objectives):
            data_iterator = data_iterators[train_idx]
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloaders[train_idx])
                data_iterators[train_idx] = data_iterator
                data = next(data_iterator)
            features, labels = data
            loss_value = loss_model(features, labels)
            losses.append(loss_value)
    return torch.mean(torch.stack(losses))    

def get_nsp_loss(dataset, model=SentenceTransformer(MODEL)):
    """Return loss from sentence transformer model using the OnlineContrastiveLoss with 
    distance metric and margin as used in sbert tutorial for duplicate questions (binary labels)

    Args:
        train_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label
        test_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label
    """

    train_examples = train_dataset_to_input_examples(dataset)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    margin = 0.5
    train_loss = losses.OnlineContrastiveLoss(
        model=model, distance_metric=distance_metric, margin=margin)
    return get_loss(train_loss, train_objectives=[(train_dataloader, train_loss)], model=model)

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


def train_sentence_transformer(train_dataset, test_dataset, model=SentenceTransformer('all-MiniLM-L6-v2')):
    """Train a sentence transformer model using the OnlineContrastiveLoss with 
    distance metric and margin as used in sbert tutorial for duplicate questions (binary labels)

    Args:
        train_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label
        test_dataset (Pandas DataFrame): A dataframe with the columns sentence_1, sentence_2 and label
    """
    train_examples = train_dataset_to_input_examples(train_dataset)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    margin = 0.5
    train_loss = losses.OnlineContrastiveLoss(
        model=model, distance_metric=distance_metric, margin=margin)
    relative_model_path = f'OnlineConstrativeLoss_{EPOCHS}Epochs-' + \
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_model_save_path = SAVE_PATH / relative_model_path
    checkpoint_path  = best_model_save_path / 'checkpoints'
    best_model_save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = len(train_dataloader) #default one in sbert
    model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=test_dataset_to_evaluator(
        test_dataset), output_path=str(best_model_save_path), save_best_model=True, epochs=EPOCHS, warmup_steps=100, show_progress_bar=True, callback=wandb_callback, checkpoint_path = str(checkpoint_path), checkpoint_save_steps=steps_per_epoch)
    log_eval_results(best_model_save_path)
    wandb.finish()

def prepare_batch_from_gan(batch):
    """
    :param batch:{'origin_text':string,'biased':string,'unbiased':string}
    """
    abstracts_list = []
    # assert False, f"batch is of type {type(batch)}) and it looks like {batch}"
    for entry in batch:
        abstracts_list.append(entry['origin_text'])
    d = {}
    for i, row in enumerate(abstracts_list):
        sentences_list = row.split('<BREAK>')
        d.update(create_labeld_dataset_from_abstract(sentences_list))
    dataset_dict = {'sentence_1': [k[0] for k in d.keys()], 'sentence_2': [
        k[1] for k in d.keys()], 'label': list(d.values())}
    dataset_df = pd.DataFrame(dataset_dict)
    dataset_df = balance_dataset(dataset_df)
    return dataset_df

def prepare_varied_batch_from_gan(batch):
    """
    :param batch:{'origin_text':string,'biased':string,'unbiased':string}
    """
    abstracts_list = []
    # assert False, f"batch is of type {type(batch)}) and it looks like {batch}"
    for entry in batch:
        abstracts_list.append(entry['origin_text'])
    d = {}
    all_sentences = []
    for i, row in enumerate(abstracts_list):
        sentences_list = row.split('<BREAK>')
        all_sentences.extend(sentences_list[::2])
        d.update(create_adjacent_pairs_from_abstract(sentences_list))
    d.update(create_negative_pairs(all_sentences))
    dataset_dict = {'sentence_1': [k[0] for k in d.keys()], 'sentence_2': [
        k[1] for k in d.keys()], 'label': list(d.values())}
    dataset_df = pd.DataFrame(dataset_dict)
    dataset_df = balance_dataset(dataset_df)
    return dataset_df

def prepare_sts_batch_from_gan(batch):
    """
    :param batch:{'origin_text':string,'biased':string,'unbiased':string}
    """
    abstracts_list = []
    # assert False, f"batch is of type {type(batch)}) and it looks like {batch}"
    for entry in batch:
        abstracts_list.append(entry['origin_text'])
    d = {}
    all_sentences = []
    for i, row in enumerate(abstracts_list):
        sentences_list = row.split('<BREAK>')
        all_sentences.extend(sentences_list[::2])
        d.update(create_same_abstract_pairs(sentences_list))
    d.update(create_different_abstract_pairs(all_sentences, d))
    dataset_dict = {'sentence_1': [k[0] for k in d.keys()], 'sentence_2': [
        k[1] for k in d.keys()], 'label': list(d.values())}
    dataset_df = pd.DataFrame(dataset_dict)
    dataset_df = balance_dataset(dataset_df)
    return dataset_df

def create_negative_pairs(sentences_list):
    dataset_dict = {}
    all_pairs_of_sentences = combinations(sentences_list, 2)
    for pair in all_pairs_of_sentences:
        dataset_dict[pair] = 0
    return dataset_dict

def create_different_abstract_pairs(sentences_list, same_abstract_pairs_dict):
    dataset_dict = {}
    all_pairs_of_sentences = combinations(sentences_list, 2)
    for pair in all_pairs_of_sentences:
        if pair not in same_abstract_pairs_dict:
            dataset_dict[pair] = 0
    return dataset_dict

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

def generate_nsp_loss_from_batch(batch, model):
    # df = pd.DataFrame(batch)
    # df.to_csv('batch.csv', index=False)
    prepared_batch = prepare_batch_from_gan(batch)
    # prepared_batch.to_csv('prepared_batch.csv', index=False)
    return get_nsp_loss(prepared_batch, model)


def run():
    wandb.init(project='train_sbert', name=f'training_{MODEL}_on_NSP_onlineContrastiveLoss_{EPOCHS}_epochs')
    train_data, test_data = prepare_data()
    train_sentence_transformer(train_data, test_data)
    # batch = [{'origin_text': 'occipital nerve block for the short-term preventive treatment of migraine a randomized double-blinded placebo-controlled study.occipital nerve on injections with corticosteroids and/or local anesthetics have been employed for the acute and preventive treatment of migraine for decades<BREAK>however to date there is no randomized placebo-controlled evidence to support the use of occipital nerve block onb for the prevention of migraine<BREAK>the objective of this article is to determine the efficacy of onb with local anesthetic and corticosteroid for the preventive treatment of migraine<BREAK>patients between <NUMBER> and <NUMBER> years old with ichd-ii-defined episodic <NUMBER> attack per week or chronic migraine modified ichd-ii as patients with <NUMBER> days with consumption of acute medications were permitted into the study were randomized to receive either <NUMBER> ml <NUMBER> bupivacaine plus <NUMBER> ml <NUMBER> mg methylprednisolone over the ipsilateral unilateral headache or bilateral bilateral headache on or <NUMBER> ml normal saline plus <NUMBER> ml <NUMBER> lidocaine without epinephrine placebo<BREAK>patients completed a one-month headache diary prior to and after the double-blind injection<BREAK>the primary outcome measure was defined as a <NUMBER> or greater reduction in the frequency of days with moderate or severe migraine headache in the four-week post-injection compared to the four-week pre-injection baseline period<BREAK>thirty-four patients received active and <NUMBER> patients received placebo treatment<BREAK>because of missing data the full analysis of <NUMBER> patients in the active and <NUMBER> patients in the placebo group was analyzed for efficacy<BREAK>in the active and placebo groups respectively the mean frequency of at least moderate mean <NUMBER> versus <NUMBER> and severe <NUMBER> versus <NUMBER> migraine days and acute medication days <NUMBER> versus <NUMBER> were not substantially different at baseline<BREAK>the percentage of patients with at least a <NUMBER> reduction in the frequency of moderate or severe headache days was <NUMBER> for both groups 10/30 vs nine of <NUMBER> δ <NUMBER> <NUMBER> ci <NUMBER> to <NUMBER>', 'biased': "lack of drug interaction between the migraine drug map0004 orally inhaled dihydroergotamine and a cyp3a4 inhibitor in humans.dihydroergotamine dhe a proven migraine treatment currently has product labeling warning against concomitant use of cyp3a4 inhibitors because of potential drug interactions<BREAK>however no reported studies of such interactions with dhe administered by any route are available<BREAK>the pharmacokinetics pk of map0004 an investigative inhaled dhe formulation were assessed in human subjects with and without cyp3a4 inhibition by ketoconazole to evaluate the potential for drug interaction elevation of dhe levels and increased adverse effects<BREAK>after map0004 alone vs map0004 plus ketoconazole the dhe maximum concentrations c max and area-under-the-curve auc 0-48 and auc 0-∞ were not statistically significantly different nor was the c max of the primary metabolite 8'-oh-dhe<BREAK>a difference in 8'-oh-dhe aucs was observed between map0004 with and without ketoconazole however the concentrations were very low<BREAK>map0004 was well tolerated after both treatments<BREAK>this study demonstrated that cyp3a4 inhibition had little to no effect on dhe pk after map0004 administration apparently because of its high systemic and low gastrointestinal bioavailability<BREAK>cyp3a4 inhibition slowed elimination of the metabolite 8'-oh-dhe but concentrations were too low to be pharmacologically relevant", 'unbiased': 'occipital nerve block for the short-term preventive treatment of migraine a randomized double-blinded placebo-controlled study.occipital nerve on injections with corticosteroids and/or local anesthetics have been employed for the acute and preventive treatment of migraine for decades<BREAK>however to date there is no randomized placebo-controlled evidence to support the use of occipital nerve block onb for the prevention of migraine<BREAK>the objective of this article is to determine the efficacy of onb with local anesthetic and corticosteroid for the preventive treatment of migraine<BREAK>patients between <NUMBER> and <NUMBER> years old with ichd-ii-defined episodic <NUMBER> attack per week or chronic migraine modified ichd-ii as patients with <NUMBER> days with consumption of acute medications were permitted into the study were randomized to receive either <NUMBER> ml <NUMBER> bupivacaine plus <NUMBER> ml <NUMBER> mg methylprednisolone over the ipsilateral unilateral headache or bilateral bilateral headache on or <NUMBER> ml normal saline plus <NUMBER> ml <NUMBER> lidocaine without epinephrine placebo<BREAK>patients completed a one-month headache diary prior to and after the double-blind injection<BREAK>the primary outcome measure was defined as a <NUMBER> or greater reduction in the frequency of days with moderate or severe migraine headache in the four-week post-injection compared to the four-week pre-injection baseline period<BREAK>thirty-four patients received active and <NUMBER> patients received placebo treatment<BREAK>because of missing data the full analysis of <NUMBER> patients in the active and <NUMBER> patients in the placebo group was analyzed for efficacy<BREAK>in the active and placebo groups respectively the mean frequency of at least moderate mean <NUMBER> versus <NUMBER> and severe <NUMBER> versus <NUMBER> migraine days and acute medication days <NUMBER> versus <NUMBER> were not substantially different at baseline<BREAK>the percentage of patients with at least a <NUMBER> reduction in the frequency of moderate or severe headache days was <NUMBER> for both groups 10/30 vs nine of <NUMBER> δ <NUMBER> <NUMBER> ci <NUMBER> to <NUMBER>'}]
    # train_data, test_data = prepare_batch_from_gan(batch)
    # train_sentence_transformer(train_data, test_data)

if __name__ == '__main__':
    run()
