import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader

from DistributionMatching.PubMed.PubMedDataSet import PubMedDataSet
from DistributionMatching import utils as project_utils
from DistributionMatching.text_utils import clean_abstracts

from DistributionMatching.utils import config
from DistributionMatching.text_utils import TextUtils

class PubMedModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.train = None
        self.test = None
        self.documents_df = None
        self.hparams.update(hparams)

    def prepare_data(self):
        # run before setup, 1 gpu
        '''
        :return:
        leaves docs with gender info only
        create "female_rate" col
        transform the model on the remaining docs and creates "topic"
        '''
        # Note - transform (bert topic inference) will take ~30 minutes, check if the df already exists
        try:
            self.documents_df = pd.read_csv(self.hparams["gender_and_topic_path"], encoding='utf8')
        except FileNotFoundError:
            documents_df = project_utils.load_abstract_PubMedData()
            # keeps docs with participants info only
            documents_df = documents_df[~documents_df['female'].isnull()]
            documents_df = documents_df[~documents_df['male'].isnull()]
            documents_df['female_rate'] = documents_df['female'] / (documents_df['female'] + documents_df['male'])
            docs = documents_df.clean_title_and_abstract.to_list()
            topics, probs = project_utils.load_topic_model().transform(docs)
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
        train_df, testing_df = train_test_split(self.documents_df, test_size=self.hparams['test_size'],random_state=42)
        test_df, val_df = train_test_split(testing_df, test_size=0.5, random_state=42)
        self.train_df = clean_abstracts(train_df)
        self.val_df = clean_abstracts(val_df)
        self.test_df = clean_abstracts(test_df)

    def setup(self, stage=None):
        # Runs on all gpus
        # Data set instances (val, train, test)

        self.train = PubMedDataSet(self.train_df, self.hparams)
        self.val = PubMedDataSet(self.val_df, self.hparams)
        self.test = PubMedDataSet(self.test_df, self.hparams)

    def train_dataloader(self):
        # data set, batch size, shuffel, workers
        return DataLoader(self.train, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=8)

