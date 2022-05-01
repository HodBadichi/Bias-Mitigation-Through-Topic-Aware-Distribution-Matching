import sys
sys.path.append('/home/mor.filo/nlp_project/')
from abc import abstractmethod
import torch
import numpy as np
import DistributionMatching.utils as project_utils


class NoahArc:
    """
    Abstract class , used to supply NoahArc API
    """

    def __init__(self,dataframe, similarity_matrix, reset_diff_topic_entries_flag, df_name, ProbabilityMatrixPath):
        self.documents_dataframe = dataframe
        self.ProbabilityMatrixPath = ProbabilityMatrixPath
        self.probability_matrix = None
        self._similarity_matrix = similarity_matrix.matrix
        self._reset_different_topic_entries_flag = reset_diff_topic_entries_flag
        self._reset_different_topics_called_flag = False

    def get_match(self, document_index):
        """
        :param document_index: The document index we need to find a matching document for
        :return: matching document index,matching document PMID to use with other dfs
        """
        if self.possible_matches(document_index) == 0:
            return None
        probabilities = self.probability_matrix[document_index]
        similar_doc_index = np.random.choice(range(0, len(self.probability_matrix)), 1, p=probabilities)
        return similar_doc_index[0]

    def possible_matches(self, document_index):
        return torch.count_nonzero(self.probability_matrix[document_index]).item()

    @abstractmethod
    def _calc_probabilities(self):
        """
        :return:Probability matrix ,differ between each metric
        """
        pass

    def _reset_different_topic_entries(self):
        self._reset_different_topics_called_flag = True
        number_of_topics = range(len(self.documents_dataframe.groupby('major_topic')))
        for idx, topic_num in enumerate(number_of_topics):
            topic_indices = list(np.where(self.documents_dataframe['major_topic'] == topic_num)[0])
            mask = np.array([True] * len(self.documents_dataframe))
            mask[topic_indices] = False
            for index in topic_indices:
                self._similarity_matrix[index][mask] = 0

    def _reset_same_bias_entries(self):
        """
            :return:
            for each row of similarity_matrix (a doc) -> keeps cross entropy score for docs from the other class
            biased doc will have scores of cross entropy with other **non biased** docs
            unbiased doc will have scores of cross entropy with other **biased** docs
        """
        biased_mask = []  # docs with women minority
        unbiased_mask = []  # docs with women majority

        for doc_index in range(len(self.documents_dataframe)):
            if project_utils.are_women_minority(doc_index, self.documents_dataframe):
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

    def _get_probability_matrix_zeros_rows(self):
        zeroed_rows_indexes = [idx for idx, row in enumerate(self._similarity_matrix) if
                               torch.count_nonzero(row).item() <= project_utils.config['minimum_documents_matches']]
        return zeroed_rows_indexes

    def _put_zeros_in_rows(self, indices, probability_matrix):
        probability_matrix[indices] = torch.zeros(len(self._similarity_matrix)).double()
        return probability_matrix
