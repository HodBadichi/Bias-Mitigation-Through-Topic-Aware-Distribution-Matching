import os
import sys
if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir,os.pardir,os.pardir))

import pandas as pd
from bertopic import BERTopic
import numpy as np
import gdown

from GAN.Utils.src.GAN_config import config
from GAN.Utils.src.TextUtils import TextUtils, CleanAbstracts
from TopicModeling.Bert.src.BertUtils import CleanText

def LoadAbstractPubMedData():
    data_directory = os.path.join(os.pardir, os.pardir, os.pardir, 'data')
    os.makedirs(data_directory, exist_ok=True)
    full_data_path = config['PubMedData']

    #   Load full dataframe
    if not os.path.exists(full_data_path):
        print("Downloading dataframe ...")
        url = 'https://drive.google.com/uc?id=1xmifhCZ4IljgjUEY73QLVPnk99Y-A04A'
        gdown.download(url, full_data_path, quiet=False)
    else:
        print("Dataframe already exists...")

    return pd.read_csv(config['PubMedData'], encoding='utf8')


def LoadTopicModel():
    models_directory = os.path.join(os.pardir,os.pardir, os.pardir, 'data')
    os.makedirs(models_directory, exist_ok=True)
    full_model_path = config['topic_model_path']

    #   Load full dataframe
    if not os.path.exists(full_model_path):
        print("Downloading dataframe ...")
        url = 'https://drive.google.com/uc?id=1K7_L-ijpb9hR43_Z0rxYcCtjBRw2x8eR'
        gdown.download(url, full_model_path, quiet=False)
    else:
        print("Dataframe already exists...")
    return BERTopic.load(config['topic_model_path'])


def GenerateGANdataframe():
    documents_df = LoadAbstractPubMedData()
    # keeps docs with participants info only
    documents_df = documents_df[~documents_df['female'].isnull()]
    documents_df = documents_df[~documents_df['male'].isnull()]
    documents_df['female_rate'] = documents_df['female'] / (documents_df['female'] + documents_df['male'])
    title_and_abstract_df = documents_df[["title_and_abstract"]]
    clean_title_and_abstract_df = CleanText(title_and_abstract_df["title_and_abstract"])
    docs = clean_title_and_abstract_df.clean_title_and_abstract_df.dropna().to_list()
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
        prob.append(1 - sum(prob))
        result_series.append(str(prob.tolist()))
    col_probs = pd.Series(result_series)
    documents_df['probs'] = col_probs
    tu = TextUtils()
    documents_df['sentences'] = documents_df['title_and_abstract'].apply(tu.SplitAbstractToSentences)
    dataframe_path = os.path.join(os.pardir, os.pardir, os.pardir, 'data', 'abstract_2005_2020_gender_and_topic.csv')
    documents_df.to_csv(dataframe_path, index=False)
    return documents_df


def SplitAndCleanDataFrame(documents_df):
    train_df = documents_df.loc[documents_df['belongs_to_group'] == 'train_dataset'].reset_index()
    test_df = documents_df.loc[documents_df['belongs_to_group'] == 'test_dataset'].reset_index()
    val_df = documents_df.loc[documents_df['belongs_to_group'] == 'val_dataset'].reset_index()
    if "broken_abstracts" not in documents_df.columns:
        train_df = CleanAbstracts(train_df)
        val_df = CleanAbstracts(val_df)
        test_df = CleanAbstracts(test_df)
    return train_df, test_df, val_df
