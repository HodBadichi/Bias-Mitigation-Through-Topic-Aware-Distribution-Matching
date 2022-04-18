from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
import pandas as pd
import json
from torch import nn
import numpy as np
import torch


class documnetPairs():
    def __init__(self, bias_by_topic = True):
        self.documents_df = self._GetDocumentsDf()
        self.similarity_matrix = self._GetSimilartyMatrix()
        self.topic_to_female_rate_mean = self._get_topic_to_female_rate_mean()
        self.bias_by_topic = bias_by_topic
        if self.bias_by_topic:
            self.ResetDiffTopicEntries()
        self.ResetDiffBiasEntries()
        self.probabilities_matrix = self._GetProbabilityMatrix()

    @staticmethod
    def _GetDocumentsDf():
            if not os.path.exists(rf'..\data\abstract_2005_2020_gender_and_topic.csv'):
                raise FileNotFoundError
            data_path = rf'..\data\abstract_2005_2020_gender_and_topic.csv'
            return pd.read_csv(data_path, encoding='utf8')

    def _GetSimilartyMatrix(self):
            if os.path.exists('ce_similarity_matrix'):
                return torch.load('ce_similarity_matrix')
            else:
                similarity_matrix=self._create_topic_ce_similarity_matrix()
                torch.save(similarity_matrix, 'ce_similarity_matrix')
                return similarity_matrix

    def _GetProbabilityMatrix(self):
            if os.path.exists('prob_matrix'):
                return torch.load('prob_matrix')
            else:
                probabilities_matrix=self._CalcProbabilities()
                torch.save(probabilities_matrix,'prob_matrix')
            return probabilities_matrix

    def cross_entropy(self, prob1, prob2, eps=1e-21):
        prob2 = torch.add(prob2, eps)
        prob2 = torch.log(prob2)
        return -prob1 @ prob2.T

    def _create_topic_ce_similarity_matrix(self):
        '''
        denote ce(i,j) : the cross entropy score of doc i topic probabilities and doc j topic probabilities
        create the cross entropy similarity matrix where each value
        similarity_matrix[i][j] = ce(i,j) + ce(j,i)
        '''
        probs = self.documents_df['probs'].apply(lambda x: json.loads(x))
        tensor_probs = torch.as_tensor(probs) # shape num_of_docs X num_of_topics (distribution for every doc)
        # probs_test = tensor_probs[[0,1,2]]
        # res = self.cross_entropy(probs_test,probs_test)  # tensor shape num_of_docs X num_of_docs
        res = self.cross_entropy(tensor_probs,tensor_probs)  # tensor shape num_of_docs X num_of_docs
        # res = res + res.T  # Make matrix symetric
        return res

    def _get_topic_to_female_rate_mean(self):
        topic_to_female_rate = self.documents_df.groupby('major_topic')['female_rate'].mean()
        return topic_to_female_rate

    def AreWomenMinority(self,document_index):
        threshold = 0.5
        female_rate = self.documents_df.iloc[document_index]["female_rate"]

        if self.bias_by_topic is True:
            topic = self.documents_df.iloc[document_index]["major_topic"]
            threshold = self.topic_to_female_rate_mean[topic]

        if(female_rate < threshold):
            return True
        else:
            return False

    def ResetDiffTopicEntries(self):
        number_of_topics = range(len(self.documents_df.groupby('major_topic')))
        for topic_num in number_of_topics:
            topic_indices = list(np.where(self.documents_df['major_topic'] == topic_num)[0])
            mask=np.array([True]*len(self.documents_df))
            mask[topic_indices] = False
            for index in topic_indices:
                self.similarity_matrix[index][mask] = 0


    def ResetDiffBiasEntries(self):
        # for each row of similarity_matrix (a doc) -> keeps cross entropy score for docs from the other class
        # biased doc will have scores of cross entropy with other **non biased** docs
        # unbiased doc will have scores of cross entropy with other **biased** docs
        biased_mask = []  # docs with women minority
        unbiased_mask = []  # docs with women majority

        for doc_index in range(len(self.documents_df)):
            if self.AreWomenMinority(doc_index) == True:
                biased_mask.append(True)
                unbiased_mask.append(False)
            else:
                biased_mask.append(False)
                unbiased_mask.append(True)

        for doc_index in range(len(self.documents_df)):
            if biased_mask[doc_index] == True:
                # the doc has women minority so the matching doc for training needs to be with women majority
                self.similarity_matrix[doc_index][biased_mask] = 0
            else:
                # the doc has women majority so the matching doc for training needs to be with women minority
                self.similarity_matrix[doc_index][unbiased_mask] = 0

    def _CalcProbabilities(self):
        prob_matrix = torch.clone(self.similarity_matrix)
        print(torch.isnan(prob_matrix).any())
        rows_sum = torch.sum(prob_matrix,1)
        print(torch.isnan(prob_matrix).any())
        arg_min = torch.argmin(rows_sum)
        print(prob_matrix[arg_min])
        prob_matrix = 1-torch.div(prob_matrix,rows_sum)
        print(torch.isnan(prob_matrix).any())
        prob_matrix[prob_matrix == 1] = 0
        print(torch.isnan(prob_matrix).any())
        rows_sum = torch.sum(prob_matrix, 1)
        print(torch.isnan(prob_matrix).any())
        prob_matrix = torch.div(prob_matrix, rows_sum)
        print(torch.isnan(prob_matrix).any())
        return prob_matrix


    def _create_embedings_similarity_matrix(self):
        '''
        denote cs(i,j) : the cosine_similarity of doc i embeddings and doc j embeddings
        create the cosine_similarity matrix where each value
        cosine_similarity_matrix[i][j] = cs(i,j)
        '''
        # cos1 = torch.nn.CosineSimilarity(dim=1)
        # output1 = cos1(docs_embeddings, tensor2)

Pairs_class = documnetPairs()
probs = Pairs_class.probabilities_matrix[2].clone()
similarity = Pairs_class.similarity_matrix[2].clone()

sum = probs.sum()

max_prob = torch.argmax(probs)
max_val = torch.max(probs)

min_ce = torch.min(similarity[torch.nonzero(similarity)])
ce_of_max_prob = similarity[max_prob.item()]
min_val = torch.min(probs)

# print(ce_of_max_prob == min_ce)


class PubMedDataSet(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        # defines len of epoch
        return len(self.dataframe)

    def __getitem__(self, index):
        result ={}
        # returns dict of 2 docs - bias\unbiased
        # {more bias: title+abs, less bias: title+abs}
        probabilities = self.docs_handler.similarity_matrix[index]
        similar_doc_index =np.random.choice(range(0, len(self.similiarty_matrix)), 1, p=probabilities)

        if self.docs_handler.AreWomenMinority(index):
            result['biased']=self.dataframe.iloc[index]["title_and_abstract"]
            result['unbiased']=self.dataframe.iloc[similar_doc_index]["title_and_abstract"]

        else:
            result['biased'] = self.dataframe.iloc[similar_doc_index]["title_and_abstract"]
            result['unbiased'] = self.dataframe.iloc[index]["title_and_abstract"]
        return result


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

