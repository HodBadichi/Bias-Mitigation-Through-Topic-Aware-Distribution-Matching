import torch
import numpy as np
from abc import abstractmethod

import DistributionMatching.Utils.Utils as project_utils
from DistributionMatching.Utils.Config import config
"""
NoahArc implementation , abstract class used to supply NoahArc interface to match between 'biased' and 'unbiased' documents
according to a certain similarity metric.
"""


class NoahArc:
    def __init__(self, dataframe, similarity_matrix, reset_diff_topic_entries_flag, df_name):
        """
            :param dataframe:pandas dataframe
            :param similarity_matrix: SimilarityMatrix class, holds the similarity between all the documents
            :param reset_diff_topic_entries_flag:Bool, whether to allow or not allow matches between documents from common topic
            :param df_name: string, 'train' or 'test'
            :return:None
        """
        self.documents_dataframe = dataframe
        self.probability_matrix = None
        self._similarity_matrix = similarity_matrix.matrix
        self._reset_different_topic_entries_flag = reset_diff_topic_entries_flag
        self.df_name = df_name

    def GetMatch(self, document_index):
        """
        :param document_index: The document index we need to find a matching document for
        :return: matching document index
        """
        if self.PossibleMatches(document_index) == 0:
            return None
        probabilities = self.probability_matrix[document_index]
        similar_doc_index = np.random.choice(range(0, len(self.probability_matrix)), 1, p=probabilities)[0]
        return similar_doc_index

    def PossibleMatches(self, document_index):
        """
        :param document_index: Count the number of possible matches for document 'document_index'
        """
        return torch.count_nonzero(self.probability_matrix[document_index]).item()

    @abstractmethod
    def _CalcProbabilities(self):
        """
        :return:Probability matrix ,differ between each metric
        """
        pass

    def _ResetDifferentTopicEntries(self):
        """
        Reset in each row all the entries which have common topics, for example:
        in the i`th row the j`th entry will be set to zero if the i and j documents share the same topic.
        """
        number_of_topics = range(len(self.documents_dataframe.groupby('major_topic')))
        for idx, topic_num in enumerate(number_of_topics):
            topic_indices = list(np.where(self.documents_dataframe['major_topic'] == topic_num)[0])
            mask = np.array([True] * len(self.documents_dataframe))
            mask[topic_indices] = False
            for index in topic_indices:
                self._similarity_matrix[index][mask] = 0

    def _ResetSameBiasEntries(self, bias_by_topic):
        """
            :return:
            for each row of similarity_matrix (a doc) -> keeps cross entropy score for docs from the other class
            biased doc will have scores of cross entropy with other **non biased** docs
            unbiased doc will have scores of cross entropy with other **biased** docs
        """
        biased_mask = []  # docs with women minority
        unbiased_mask = []  # docs with women majority

        for doc_index in range(len(self.documents_dataframe)):
            if project_utils.AreWomenMinority(doc_index, self.documents_dataframe, bias_by_topic):
                biased_mask.append(True)
                unbiased_mask.append(False)
            else:
                biased_mask.append(False)
                unbiased_mask.append(True)

        for doc_index in range(len(self.documents_dataframe)):
            if biased_mask[doc_index]:
                # the doc has women minority so the matching doc for training needs to be with women majority
                self._similarity_matrix[doc_index][biased_mask] = 0
            else:
                # the doc has women majority so the matching doc for training needs to be with women minority
                self._similarity_matrix[doc_index][unbiased_mask] = 0

    def _GetProbabilityMatrixZerosRows(self):
        zeroed_rows_indexes = [idx for idx, row in enumerate(self._similarity_matrix) if
                               torch.count_nonzero(row).item() <= config['minimum_documents_matches'][self.df_name]]
        return zeroed_rows_indexes

    def _PutZerosInRows(self, indices, probability_matrix):
        probability_matrix[indices] = torch.zeros(len(self._similarity_matrix)).double()
        return probability_matrix
