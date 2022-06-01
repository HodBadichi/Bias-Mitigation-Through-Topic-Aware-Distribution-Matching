import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader

from GAN.GANPubMed.src.PubMedDataSet import PubMedDataSet
from GAN.Utils.TextUtils import CleanAbstracts
from GAN.Utils import Utils as GAN_utils


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
            self.documents_df = GAN_utils.GenerateGANdataframe()


        # train_test_split was done once and in order to make sure we keep the same groups
        # we will use the "belong to group" column

        train_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'train_dataset']
        test_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'test_dataset']
        val_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'val_dataset']
        self.train_df = train_df.reset_index()
        self.test_df = test_df.reset_index()
        self.val_df = val_df.reset_index()
        if "broken_abstracts" not in self.documents_df.columns:
            self.train_df = CleanAbstracts(self.train_df)
            self.val_df = CleanAbstracts(self.val_df)
            self.test_df = CleanAbstracts(self.test_df)

    def setup(self, stage=None):
        # Runs on all gpus
        # Data set instances (val_dataset, train_dataset, test_dataset)
        self.train = PubMedDataSet(self.train_df, self.hparams, "train_dataset", self.hparams['SimilarityMatrixPathTrain'],
                                   self.hparams['ProbabilityMatrixPathTrain'])
        self.val = PubMedDataSet(self.val_df, self.hparams, "val_dataset", self.hparams['SimilarityMatrixPathVal'],
                                 self.hparams['ProbabilityMatrixPathVal'])
        self.test = PubMedDataSet(self.test_df, self.hparams, "test_dataset", self.hparams['SimilarityMatrixPathTest'],
                                  self.hparams['ProbabilityMatrixPathTest'])

    def train_dataloader(self):
        # data set, batch size, shuffel, workers
        return DataLoader(self.train, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=2)
