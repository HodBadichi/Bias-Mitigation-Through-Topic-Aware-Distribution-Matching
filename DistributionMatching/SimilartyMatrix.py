import torch
import pandas as pd
import json
import numpy as np
import DistributionMatching.utils as project_utils
from sklearn.metrics.pairwise import cosine_similarity
import os

class SimilartyMatrix:
    def __init__(self, documents_dataframe, similarty_type, bias_by_topic=True):
        """
            :param documents_dataframe
            :param bias_by_Topic True= bias threshold is the topic`s woman_rate mean, False =bias threshold is 0.5
            :param similarty_type : "cross_entropy" = topic probs cross entropy, "cosine_similarity" = bert doc embeddings cosine_similarity
        """
        self.bias_by_topic = bias_by_topic
        try:
            self.documents_dataframe = pd.read_csv("SimMatrixDf", encoding='utf8')
            self.matrix = torch.load("SimMatrixTensor")
        except FileNotFoundError:
            print("Couldn`t find Similarty matrix on disk , Calculating...")
            self.documents_dataframe = documents_dataframe
            if similarty_type == "cross_entropy":
                self.matrix = self._CalcCESimilarties()
            elif similarty_type == "cosine_similarity":
                self.matrix = self._CalcCSSimilarties()
            self._ResetDiffTopicEntries_called_flag = False
            self._ResetSameBiasEntries_called_flag = False

    def _CalcCESimilarties(self):
        '''
            denote ce(i,j) : the cross entropy score of doc i topic probabilities and doc j topic probabilities
            create the cross entropy similarity matrix where each value
            similarity_matrix[i][j] = ce(i,j)
        '''
        probs = self.documents_dataframe['probs'].apply(lambda x: json.loads(x))
        tensor_probs = torch.as_tensor(probs)  # shape num_of_docs X num_of_topics (distribution for every doc)
        # probs_test = tensor_probs[[0,1,2]]
        # res = self.cross_entropy(probs_test,probs_test)  # tensor shape num_of_docs X num_of_docs
        res = self._CrossEntropy(tensor_probs, tensor_probs)  # tensor shape num_of_docs X num_of_docs
        # res = res + res.T  # Make matrix symetric
        return res


    def _CalcCSSimilarties(self, embeddings):
        '''
            denote ce(i,j) : the cosine similarity of doc i embeddings and doc j embeddings
            create the cosine similarity similarity matrix where each value
            similarity_matrix[i][j] = (i embeddings)dot(j embeddings)/max(l2_norm(i embeddings)*l2_norm(j embeddings),eps)
        '''
        return cosine_similarity(embeddings, embeddings)

    def SaveMatrix(self):
        torch.save(self.matrix, "SimMatrixTensor")
        self.documents_dataframe.to_csv( "SimMatrixDf")

    def ResetDiffTopicEntries(self):
        self._ResetDiffTopicEntries_called_flag = True
        number_of_topics = range(len(self.documents_dataframe.groupby('major_topic')))
        for topic_num in number_of_topics:
            topic_indices = list(np.where(self.documents_dataframe['major_topic'] == topic_num)[0])
            mask = np.array([True] * len(self.documents_dataframe))
            mask[topic_indices] = False
            for index in topic_indices:
                self.matrix[index][mask] = 0

    def ResetSameBiasEntries(self):
        '''
            :return:
            for each row of similarity_matrix (a doc) -> keeps cross entropy score for docs from the other class
            biased doc will have scores of cross entropy with other **non biased** docs
            unbiased doc will have scores of cross entropy with other **biased** docs
        '''
        self._ResetSameBiasEntries_called_flag = True
        biased_mask = []  # docs with women minority
        unbiased_mask = []  # docs with women majority

        for doc_index in range(len(self.documents_dataframe)):
            if project_utils.AreWomenMinority(doc_index,self.documents_dataframe) == True:
                biased_mask.append(True)
                unbiased_mask.append(False)
            else:
                biased_mask.append(False)
                unbiased_mask.append(True)

        for doc_index in range(len(self.documents_dataframe)):
            if biased_mask[doc_index] == True:
                # the doc has women minority so the matching doc for training needs to be with women majority
                self.matrix[doc_index][biased_mask] = 0
            else:
                # the doc has women majority so the matching doc for training needs to be with women minority
                self.matrix[doc_index][unbiased_mask] = 0

    def _CrossEntropy(self, prob1, prob2, eps=1e-21):
        '''
            :param prob1: tensor that represent probability
            :param prob2: tensor that represent probability
            :param eps: to avoid log(0)
            :return:cross entropy between prob1 and prob2 by the formula: -sum(prob1*log(prob2))
        '''
        prob2 = torch.add(prob2, eps)
        prob2 = torch.log(prob2)
        return -prob1 @ prob2.T

    def DropUnWantedDocsAndReset(self):
        """
            Drop rows from the similarity matrix for 2 reasons:
            1. The similarity matrix is later transformed to probability, so we don't want null rows
            2. We want to define the "NOT_ENOUGH_PAIRS_THRESHOLD" for topics and pairs that are homogeneous
            :param ResetDiffTopics: reset flag for the new CE matrix after droppping unwanted lines
            :param ResetSameBias: reset flag for the new CE matrix after droppping unwanted lines
            :return: Calculate new similarity matrix after dropping the null\sparse rows
        """
        NOT_ENOUGH_PAIRS_THRESHOLD = 1
        rows_to_drop = []
        frame_size = len(self.documents_dataframe)
        for document_index in range(frame_size):
            MatchingDocumentsNum = torch.count_nonzero(self.matrix[document_index]).item()
            if MatchingDocumentsNum < NOT_ENOUGH_PAIRS_THRESHOLD:
                rows_to_drop.append(document_index)
        self.documents_dataframe = self.documents_dataframe.drop(rows_to_drop)
        self.documents_dataframe = self.documents_dataframe.reset_index()
        self.matrix = self._CalcCESimilarties()
        if self.ResetDiffTopics_called_flag == True:
            self.ResetDiffTopicEntries()
        if self.ResetSameBiasEntries_called_flag == True:
            self.ResetSameBiasEntries()

if __name__=='__main__':
    df = pd.read_csv(project_utils.DATAFRAME_PATH,
                     encoding='utf8')
    matrix = SimilartyMatrix(df)
    print(len(matrix.documents_df))
    matrix.ResetSameBiasEntries()
    matrix.ResetDiffTopicEntries()
    print(torch.count_nonzero(matrix.matrix[282]).item())
    matrix.DropUnWantedDocsAndReset()
    matrix.SaveMatrix()
    print(len(matrix.documents_df))
