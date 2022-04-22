from DistributionMatching.NoahArc.NoahArc import NoahArc
import torch


class NoahArcCE(NoahArc):
    def __init__(self, reset_different_topic_entries_flag, similarity_matrix=None):
        super().__init__(reset_different_topic_entries_flag, similarity_matrix)
        self._reset_same_bias_entries()
        if reset_different_topic_entries_flag:
            self._reset_different_topic_entries()
        self._drop_unwanted_document_rows(similarity_metric="cross_entropy")
        self.probability_matrix = self._calc_probabilities()

    def _calc_probabilities(self):
        """
                The formula for each similarity entry X:
                X_TEMP =(1-(X/sum(X_row_entries)) - low cross entropy means more similar so we want to flip the scores:
                the lower the cross entropy score - the higher the prob will be
                new_row_entry = X_TEMP/sum(X_temp_row_entries)
                return: new_row_entry
        """
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
        return probability_matrix
