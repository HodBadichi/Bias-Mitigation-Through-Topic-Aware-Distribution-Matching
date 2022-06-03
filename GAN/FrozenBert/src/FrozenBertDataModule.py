import os
import sys
sys.path.append(os.path.join(os.pardir,os.pardir,os.pardir))
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from GAN.Utils.src import Utils as GAN_utils
from GAN.FrozenBert.src.FrozenBertDataSet import FrozenBertDataSet

"""
Data model implementation.
A batch consists with cleaned title and abstract for bert (using CleanAbstracts)
"""


class FrozenBertDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.test_df = None
        self.val_df = None
        self.train_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.documents_df = None

    def prepare_data(self):
        """
        Creates train,test,validation data frames.
        In case the needed data does not exist it generates it using 'GAN_UTILS'
        """

        dataframe_path = os.path.join(
            os.pardir,
            os.pardir,
            os.pardir,
            'data',
            'abstract_2005_2020_gender_and_topic.csv'
        )
        if not os.path.exists(dataframe_path):
            GAN_utils.GenerateGANdataframe()
        self.documents_df = pd.read_csv(dataframe_path, encoding='utf8')
        # train_test_split was done once and in order to make sure we keep the same groups
        # we will use the "belong to group" column
        self.train_df, self.test_df, self.val_df = GAN_utils.SplitAndCleanDataFrame(self.documents_df)

    def setup(self, stage=None):
        self.train_dataset = FrozenBertDataSet(self.train_df)
        self.val_dataset = FrozenBertDataSet(self.val_df)
        self.test_dataset = FrozenBertDataSet(self.test_df)

    def train_dataloader(self):
        # data set, batch size, shuffle, workers
        return DataLoader(self.train_dataset, shuffle=True, batch_size=16, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=16, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=16, num_workers=8)
