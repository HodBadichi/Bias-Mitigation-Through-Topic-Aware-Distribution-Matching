import sys
sys.path.append('/home/mor.filo/nlp_project/')
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from GAN.PubMed.text_utils import clean_abstracts, TextUtils
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl

"""
Dataset and data model implementation.
A batch consists with cleaned title and abstract for bert (using clean_abstracts)
"""


class PubMedDataSetForBert(Dataset):
    def __init__(self, documents_dataframe):
        self.df = documents_dataframe
        self.tu = TextUtils()

    def __len__(self):
        # defines len of epoch
        return len(self.df)

    def __getitem__(self, index):
        return {'text': self.df.iloc[index]["broken_abstracts"]}


class PubMedModuleForBert(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.test_df = None
        self.val_df = None
        self.train_df = None
        self.train = None
        self.val = None
        self.test = None
        self.documents_df = None

    def prepare_data(self):
        self.documents_df = pd.read_csv(rf'../../data/abstract_2005_2020_gender_and_topic.csv', encoding='utf8')
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
        self.train = PubMedDataSetForBert(self.train_df)
        self.val = PubMedDataSetForBert(self.val_df)
        self.test = PubMedDataSetForBert(self.test_df)

    def train_dataloader(self):
        # data set, batch size, shuffle, workers
        return DataLoader(self.train, shuffle=True, batch_size=16, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=16, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=16, num_workers=8)