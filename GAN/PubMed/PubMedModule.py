import sys
import os

if os.name !='nt':
    sys.path.append('/home/mor.filo/nlp_project/')

import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from GAN.PubMed.PubMedDataSet import PubMedDataSet
from DistributionMatching.Utils import Utils as project_utils
from GAN.PubMed.text_utils import clean_abstracts
from GAN.PubMed.text_utils import TextUtils


class PubMedModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.train = None
        self.test = None
        self.documents_df = None
        self.hparams.update(hparams)

    def prepare_data(self):
        # run before setup, 1 gpu
        """
        :return:
        leaves docs with gender info only
        Create "female_rate" col
        transform the model on the remaining docs and creates "topic"
        """
        # Note - transform (bert topic inference) will take ~30 minutes, check if the df already exists
        try:
            self.documents_df = pd.read_csv(self.hparams["gender_and_topic_path"], encoding='utf8')
        except FileNotFoundError:
            documents_df = project_utils.LoadAbstractPubMedData()
            # keeps docs with participants info only
            documents_df = documents_df[~documents_df['female'].isnull()]
            documents_df = documents_df[~documents_df['male'].isnull()]
            documents_df['female_rate'] = documents_df['female'] / (documents_df['female'] + documents_df['male'])
            docs = documents_df.clean_title_and_abstract.to_list()
            topics, probs = project_utils.LoadTopicModel().transform(docs)
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
            documents_df['sentences'] = documents_df['title_and_abstract'].apply(tu.split_abstract_to_sentences)
            self.documents_df = documents_df

        # train_test_split was done once and in order to make sure we keep the same groups
        # we will use the "belong to group" column


        train_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'train']
        test_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'test']
        val_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'val']
        self.train_df = train_df.reset_index()
        self.test_df = test_df.reset_index()
        self.val_df = val_df.reset_index()
        if "broken_abstracts" not in self.documents_df.columns:
            self.train_df = clean_abstracts(self.train_df)
            self.val_df = clean_abstracts(self.val_df)
            self.test_df = clean_abstracts(self.test_df)

    def setup(self, stage=None):
        # Runs on all gpus
        # Data set instances (val, train, test)
        self.train = PubMedDataSet(self.train_df, self.hparams, "train", self.hparams['SimilarityMatrixPathTrain'], self.hparams['ProbabilityMatrixPathTrain'])
        self.val = PubMedDataSet(self.val_df, self.hparams, "val", self.hparams['SimilarityMatrixPathVal'], self.hparams['ProbabilityMatrixPathVal'])
        self.test = PubMedDataSet(self.test_df, self.hparams, "test", self.hparams['SimilarityMatrixPathTest'], self.hparams['ProbabilityMatrixPathTest'])



    def train_dataloader(self):
        # data set, batch size, shuffel, workers
        return DataLoader(self.train, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)

