import sys
sys.path.append('/home/mor.filo/nlp_project/')
from DistributionMatching.NoahArc.NoahArc import NoahArc
import torch
import os

class NoahArcCE(NoahArc):
    def __init__(self,dataframe, similarity_matrix, reset_different_topic_entries_flag, ProbabilityMatrixPath):
        super().__init__(dataframe, similarity_matrix, reset_different_topic_entries_flag, ProbabilityMatrixPath)
        if (os.path.isfile(self.ProbabilityMatrixPath)):
            self.probability_matrix = torch.load(self.ProbabilityMatrixPath)
        else:
            self._reset_same_bias_entries()
            if reset_different_topic_entries_flag:
                self._reset_different_topic_entries()
            self.probability_matrix = self._calc_probabilities()

    def _calc_probabilities(self):
        """
                The formula for each similarity entry X:
                X_TEMP =(1-(X/sum(X_row_entries)) - low cross entropy means more similar so we want to flip the scores:
                the lower the cross entropy score - the higher the prob will be
                new_row_entry = X_TEMP/sum(X_temp_row_entries)
                return: new_row_entry
        """
        zeroed_rows_indexes = self._get_probability_matrix_zeros_rows()
        # Transform float32 Tensor to float64 Tensor to prevent numeric errors (rounding to 0 in small values)
        probability_matrix = self._similarity_matrix.double()
        rows_sum_vector = torch.sum(probability_matrix, 1).reshape((-1, 1))
        probability_matrix = torch.div(probability_matrix, rows_sum_vector)
        # No need to flip (1-x) zero and one values
        probability_matrix = torch.where((probability_matrix != 0) & (probability_matrix != 1),
                                         1 - probability_matrix,
                                         probability_matrix)
        rows_sum_vector = torch.sum(probability_matrix, 1).reshape(-1, 1)
        probability_matrix = torch.div(probability_matrix, rows_sum_vector)
        probability_matrix = self._put_zeros_in_rows(zeroed_rows_indexes, probability_matrix)
        return probability_matrix
