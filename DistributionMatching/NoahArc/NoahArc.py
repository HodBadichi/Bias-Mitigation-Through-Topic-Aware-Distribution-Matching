from abc import abstractmethod
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory
import torch
import numpy as np
import DistributionMatching.utils as project_utils


class NoahArc:
    """
    Abstract class , used to supply NoahArc API
    """
    def __init__(self, reset_diff_topic_entries_flag, similarity_matrix):
        self.documents_dataframe = similarity_matrix.documents_dataframe
        self.probability_matrix = None
        self._similarity_matrix = similarity_matrix.matrix
        self._reset_different_topic_entries_flag = reset_diff_topic_entries_flag
        self._reset_different_topics_called_flag = False

    def get_match(self, document_index):
        """
        :param document_index: The document index we need to find a matching document for
        :return: matching document index,matching document PMID to use with other dfs
        """
        probabilities = self.probability_matrix[document_index]
        similar_doc_index = np.random.choice(range(0, len(self.probability_matrix)), 1, p=probabilities)
        return similar_doc_index[0], self.documents_dataframe['PMID'][similar_doc_index]

    @abstractmethod
    def _calc_probabilities(self):
        """
        :return:Probability matrix ,differ between each metric
        """
        pass

    def _reset_different_topic_entries(self):
        self._reset_different_topics_called_flag = True
        number_of_topics = range(len(self.documents_dataframe.groupby('major_topic')))
        for topic_num in number_of_topics:
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

    def _drop_unwanted_document_rows(self, similarity_metric):
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
            matching_documents_num = torch.count_nonzero(self._similarity_matrix[document_index]).item()
            if matching_documents_num < NOT_ENOUGH_PAIRS_THRESHOLD:
                rows_to_drop.append(document_index)
        self.documents_dataframe = self.documents_dataframe.drop(rows_to_drop)
        self.documents_dataframe = self.documents_dataframe.reset_index()
        self._similarity_matrix = SimilarityMatrixFactory.create(self.documents_dataframe,
                                                                 similarity_metric).matrix
        self._reset_same_bias_entries()
        if self._reset_different_topics_called_flag:
            self._reset_different_topic_entries()
