import sys
sys.path.append('/home/mor.filo/nlp_project/')
from DistributionMatching.NoahArc.NoahArc import NoahArc
import torch


class NoahArcCS(NoahArc):
    def __init__(self,dataframe, reset_same_topic_entries, similarity_matrix=None):
        super().__init__(dataframe, reset_same_topic_entries, similarity_matrix)
        self.reset_same_bias_entries()
        self.probability_matrix = self._calc_probabilities()

    def _calc_probabilities(self):
        probability_matrix = self._similarity_matrix.double()
        rows_sum_vector = torch.sum(probability_matrix, 1).reshape((-1, 1))
        probability_matrix = torch.div(probability_matrix, rows_sum_vector)
        torch.save(self.probability_matrix, "prob_matrix_with_BERTopic_clean")
        return probability_matrix