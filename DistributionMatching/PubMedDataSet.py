from torch.utils.data import Dataset, DataLoader
from DistributionMatching.NoahArc import NoahArc
from sklearn.model_selection import train_test_split
from DistributionMatching.SimilartyMatrix import SimilartyMatrix
import utils as project_utils
import pytorch_lightning as pl
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from DistributionMatching.utils import config
from torch import nn
import numpy as np
import torch


class PubMedDataSet(Dataset):
    def __init__(self,documents_dataframe):
        self.Matcher = PubMedDataSet._buildNoahArc(documents_dataframe, similarty_type=config['similarty_metric'])
        self.modified_document_df = self.Matcher.documents_dataframe

    def __len__(self):
        # defines len of epoch
        return len(self.modified_document_df)

    def __getitem__(self, index):
        result ={}
        document_df = self.modified_document_df

        similar_doc_index = self.Matcher.GetMatch(index)[0]
        if project_utils.AreWomenMinority(index,self.Matcher.documents_dataframe):
            result['biased']=(document_df.iloc[index]["title_and_abstract"],
                              document_df.iloc[index]["female_rate"],
                              document_df.iloc[index]["major_topic"])

            result['unbiased']=(document_df.iloc[similar_doc_index]["title_and_abstract"],
                                document_df.iloc[similar_doc_index]["female_rate"],
                                document_df.iloc[similar_doc_index]["major_topic"])
        else:
            result['biased'] = (document_df.iloc[similar_doc_index]["title_and_abstract"],
                                document_df.iloc[similar_doc_index]["female_rate"],
                                document_df.iloc[similar_doc_index]["major_topic"])

            result['unbiased'] = (document_df.iloc[index]["title_and_abstract"],
                                  document_df.iloc[index]["female_rate"],
                                  document_df.iloc[index]["major_topic"])
        return result

    @staticmethod
    def _buildNoahArc(dataframe, similarty_type):
        if NoahArc.IsNoahSaved():
            return NoahArc(similarty_type)
        else:
            SimMatrix = SimilartyMatrix(dataframe, similarty_type)
            # todo modify the next 3 resets by word embedding needs
            SimMatrix.ResetSameBiasEntries()
            SimMatrix.ResetDiffTopicEntries()
            SimMatrix.DropUnWantedDocsAndReset()
            SimMatrix.SaveMatrix()
            Arc = NoahArc(similarty_type, SimMatrix)
            Arc.Save()
        return Arc


class PubMedModule(pl.LightningDataModule):
    def __init__(self):
        pass

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
            self.documents_df = pd.read_csv(config['data']['gender_and_topic_path'], encoding='utf8')
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
            self.documents_df = documents_df


    def setup(self):
        # runs on all gpus
        # data set instanses (val, train, test)
        train_df, test_df = train_test_split(self.documents_df, test_size=config['test_size'])
        # todo split_abstracts_to_sentences_df ? all the text utils
        self.train = PubMedDataSet(train_df)
        self.test = PubMedDataSet(test_df)

    def train_dataloader(self):
        # data set, batch size, shuffel, workers
        return DataLoader(self.train, shuffle=True, batch_size=config['train']['batch_size'], num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=True, batch_size=config['test']['batch_size'], num_workers=1)

    def val_dataloader(self):
        pass


if __name__ == '__main__':
    dl = PubMedModule()
    dl.prepare_data()
    dl.setup()
    for batch in dl.train_dataloader():
        print(type(batch))