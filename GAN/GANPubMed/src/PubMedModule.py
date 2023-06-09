import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader

from GAN.GANPubMed.src.PubMedDataSet import PubMedDataSet
from GAN.Utils.src import Utils as GAN_utils


class PubMedModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.documents_df = None
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

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
            self.documents_df = pd.read_csv(self.hparams["gender_and_topic_path"]+".csv", encoding='utf8')
            self.train_df = pd.read_csv(self.hparams["gender_and_topic_path"]+"_train.csv", encoding='utf8')
            self.val_df = pd.read_csv(self.hparams["gender_and_topic_path"]+"_val.csv", encoding='utf8')
            self.test_df = pd.read_csv(self.hparams["gender_and_topic_path"]+"_test.csv", encoding='utf8')
            
        except FileNotFoundError:
            print("are we here?")
            self.documents_df,self.train_df, self.val_df, self.test_df = GAN_utils.GenerateGANdataframe()
            self.documents_df.to_csv(self.hparams["gender_and_topic_path"]+".csv", encoding='utf8')
            self.train_df.to_csv(self.hparams["gender_and_topic_path"]+"_train.csv", encoding='utf8')
            self.val_df.to_csv(self.hparams["gender_and_topic_path"]+"_val.csv", encoding='utf8')
            self.test_df.to_csv(self.hparams["gender_and_topic_path"]+"_test.csv", encoding='utf8')

            
        # train_test_split was done once and in order to make sure we keep the same groups
        # we will use the "belong to group" column
        self.train_df, self.test_df, self.val_df = GAN_utils.SplitAndCleanDataFrame(self.documents_df)

    def setup(self, stage=None):
        # Runs on all gpus
        # Data set instances (val_dataset, train_dataset, test_dataset)
        self.train_dataset = PubMedDataSet(self.train_df, self.hparams, "train_dataset")
        self.val_dataset = PubMedDataSet(self.val_df, self.hparams, "val_dataset")
        self.test_dataset = PubMedDataSet(self.test_df, self.hparams, "test_dataset")

    def train_dataloader(self):
        # data set, batch size, shuffel, workers
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)
