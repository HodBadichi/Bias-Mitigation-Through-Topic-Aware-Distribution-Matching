from DistributionMatching.NoahArc.NoahArc import NoahArc
import torch


class NoahArcCS(NoahArc):
    def __init__(self, reset_same_topic_entries, similarity_matrix=None):
        super.init(reset_same_topic_entries, similarity_matrix)
        self.reset_same_bias_entries()
        if reset_same_topic_entries:
            self.reset_different_topic_entries()
        self.probability_matrix = self._calc_probabilities()

    def _calc_probabilities(self):
        probability_matrix = self._similarity_matrix.double()
        rows_sum_vector = torch.sum(probability_matrix, 1).reshape((-1, 1))
        probability_matrix = torch.div(probability_matrix, rows_sum_vector)
        return probability_matrix
