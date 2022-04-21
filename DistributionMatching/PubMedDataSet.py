from torch.utils.data import Dataset, DataLoader
from DistributionMatching.NoahArc import NoahArc
from DistributionMatching.SimilartyMatrix import SimilartyMatrix
import utils as project_utils
import pytorch_lightning as pl
import os
import pandas as pd
import json
from torch import nn
import numpy as np
import torch



class PubMedDataSet(Dataset):
    def __init__(self,documents_dataframe):
        self.Matcher = PubMedDataSet._buildNoahArc(documents_dataframe)
        self.modified_document_df = self.Matcher.document_df

    def __len__(self):
        # defines len of epoch
        return len(self.modified_document_df)

    def __getitem__(self, index):
        '''
        :param index: index of document to find its match for training
        :return: returns dict of 2 docs - bias\unbiased
        {more bias: title+abs, less bias: title+abs}
        '''
        result ={}
        similar_doc_index = self.Matcher.GetMatch(index)
        if project_utils.AreWomenMinority(index,self.Matcher.document_df):
            result['biased']=self.modified_document_df.iloc[index]["title_and_abstract"]
            result['unbiased']=self.modified_document_df.iloc[similar_doc_index]["title_and_abstract"]
        else:
            result['biased'] = self.modified_document_df.iloc[similar_doc_index]["title_and_abstract"]
            result['unbiased'] = self.modified_document_df.iloc[index]["title_and_abstract"]
        return result

    @staticmethod
    def _buildNoahArc(dataframe):
        if NoahArc.IsNoahSaved():
            return NoahArc()
        else:
            SimMatrix = SimilartyMatrix(dataframe)
            SimMatrix.ResetSameBiasEntries()
            SimMatrix.ResetDiffTopicEntries()
            SimMatrix.DropUnWantedDocsAndReset()
            SimMatrix.SaveMatrix()
            Arc = NoahArc(SimMatrix)
            Arc.Save()
        return Arc





# class PubMedModule(pl.LightningDataModule):
#     def __init__(self):
#         pass
#
#     def prepare_data(self):
#         # run before setup, 1 gpu
#         pass
#
#     def setup(self):
#         # runs on all gpus
#         # data set instanses (val, train, test)
#         pass
#
#     def train_dataloader(self):
#         # data set, batch size, shuffel, workers
#         pass
#
#     def test_dataloader(self):
#         pass
#
#     def val_dataloader(self):
#         pass

