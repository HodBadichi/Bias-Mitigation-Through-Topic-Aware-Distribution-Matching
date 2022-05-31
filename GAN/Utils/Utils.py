import pandas as pd
from bertopic import BERTopic
from GAN_config import hparams


def LoadAbstractPubMedData():
    return pd.read_csv(hparams['PubMedData'], encoding='utf8')


def LoadTopicModel():
    return BERTopic.load(hparams['topic_model_path'])
