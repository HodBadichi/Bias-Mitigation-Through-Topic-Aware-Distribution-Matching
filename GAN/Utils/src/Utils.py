import os
import sys

if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

import pandas as pd
from bertopic import BERTopic
import numpy as np
import gdown
import pathlib

from GAN.Utils.src.GAN_config import config
from GAN.Utils.src.TextUtils import TextUtils, CleanAbstracts
from TopicModeling.Bert.src.BertUtils import CleanText


def LoadAbstractPubMedData():
    data_directory = pathlib.Path(__file__).parent.resolve().parents[2] / 'data'
    os.makedirs(data_directory, exist_ok=True)
    full_data_path = config['PubMedData']

    #   Load full dataframe
    if not os.path.exists(full_data_path):
        print("Downloading dataframe ...")
        url = 'https://drive.google.com/u/0/uc?id=1bd7mcDnnXcXHKSHHqAUWHxJX-QREQJ_O'
        gdown.download(url, full_data_path, quiet=False)
    else:
        print("Dataframe already exists...")
    return pd.read_csv(config['PubMedData'], encoding='utf8')


def LoadTopicModel():
    models_directory = pathlib.Path(__file__).parent.resolve().parents[2] / 'data'
    os.makedirs(models_directory, exist_ok=True)
    full_model_path = config['topic_model_path']

    #   Load full dataframe
    if not os.path.exists(full_model_path):
        print("Downloading model ...")
        url = 'https://drive.google.com/u/0/uc?id=1mxdyCfzqNgBj0l6VtgQsFlCGtND_-TGk'
        gdown.download(url, full_model_path, quiet=False)
    else:
        print("Model already exists...")
    return BERTopic.load(config['topic_model_path'])


def GenerateGANdataframe():
    documents_df = LoadAbstractPubMedData()
    # keeps docs with participants info only
    documents_df = documents_df[~documents_df['female'].isnull()]
    documents_df = documents_df[~documents_df['male'].isnull()]
    documents_df['female_rate'] = documents_df['female'] / (documents_df['female'] + documents_df['male'])
    clean_title_and_abstract_df = CleanText(documents_df["title_and_abstract"])
    docs = clean_title_and_abstract_df.dropna().to_list()
    topics, probs = LoadTopicModel().transform(docs)
    col_topics = pd.Series(topics)
    # get topics
    documents_df['topic_with_outlier_topic'] = col_topics
    # convert "-1" topic to second best
    main_topic = [np.argmax(item) for item in probs]
    documents_df['major_topic'] = main_topic
    # get probs and save as str
    result_series = []
    for prob in probs:
        # add topic -1 prob - since probs sum up to the probability of not being outlier
        prob = prob.tolist()
        prob.append(1 - sum(prob))
        result_series.append(str(prob))
    col_probs = pd.Series(result_series)
    documents_df['probs'] = col_probs
    tu = TextUtils()
    documents_df['sentences'] = documents_df['title_and_abstract'].apply(tu.SplitAbstractToSentences)
    dataframe_path =  pathlib.Path(__file__).parent.resolve().parents[2] / 'data'/ 'abstract_2005_2020_gender_and_topic.csv'
    if "broken_abstracts" not in documents_df.columns:
        documents_df = CleanAbstracts(documents_df)
    train_df = documents_df.loc[documents_df['belongs_to_group'] == 'train'].reset_index()
    test_df = documents_df.loc[documents_df['belongs_to_group'] == 'test'].reset_index()
    val_df = documents_df.loc[documents_df['belongs_to_group'] == 'val'].reset_index()

    # documents_df.to_csv(dataframe_path, index=False)

    return documents_df,train_df,val_df,test_df


def SplitAndCleanDataFrame(documents_df):
    train_df = documents_df.loc[documents_df['belongs_to_group'] == 'train'].reset_index()
    test_df = documents_df.loc[documents_df['belongs_to_group'] == 'test'].reset_index()
    val_df = documents_df.loc[documents_df['belongs_to_group'] == 'val'].reset_index()
    return train_df, test_df, val_df
