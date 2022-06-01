import os

import pandas as pd
from bertopic import BERTopic
import numpy as np

from GAN_config import config
from GAN.Utils.TextUtils import TextUtils, CleanAbstracts


def LoadAbstractPubMedData():
    return pd.read_csv(config['PubMedData'], encoding='utf8')


def LoadTopicModel():
    return BERTopic.load(config['topic_model_path'])


def GenerateGANdataframe():
    documents_df = LoadAbstractPubMedData()
    # keeps docs with participants info only
    documents_df = documents_df[~documents_df['female'].isnull()]
    documents_df = documents_df[~documents_df['male'].isnull()]
    documents_df['female_rate'] = documents_df['female'] / (documents_df['female'] + documents_df['male'])
    docs = documents_df.clean_title_and_abstract.to_list()
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
