import sys
sys.path.append('/home/mor.filo/nlp_project/')
from DistributionMatching.NoahArc.NoahArc import NoahArc
import torch
import os

class NoahArcCS(NoahArc):
    def __init__(self,dataframe, similarity_matrix, reset_different_topic_entries_flag, ProbabilityMatrixPath):
        super().__init__(dataframe, similarity_matrix, reset_different_topic_entries_flag, ProbabilityMatrixPath)
        if (os.path.isfile(self.ProbabilityMatrixPath)):
            self.probability_matrix = torch.load(self.ProbabilityMatrixPath)
        else:
            self._reset_same_bias_entries()
            self.probability_matrix = self._calc_probabilities()

    def _calc_probabilities(self):
        probability_matrix = self._similarity_matrix.double()
        rows_sum_vector = torch.sum(probability_matrix, 1).reshape((-1, 1))
        probability_matrix = torch.div(probability_matrix, rows_sum_vector)
        return probability_matrix