import torch
import os

from DistributionMatching.NoahArc.src.NoahArc import NoahArc


"""NoahArcCE implementation
inherits from 'NoahArc' , based on Cross entropy metric  
"""


class NoahArcCE(NoahArc):
    def __init__(
            self,
            dataframe,
            similarity_matrix,
            reset_different_topic_entries_flag,
            df_name,
            ProbabilityMatrixPath
    ):
        """
            :param dataframe:pandas dataframe
            :param similarity_matrix: SimilarityMatrix class, holds the similarity between all the documents
            :param reset_diff_topic_entries_flag:Bool, whether to allow or not allow matches between documents from common topic
            :param df_name: string, 'train' or 'test'
            :param ProbabilityMatrixPath: Path, in case the probability matrix already exists
            :return:None
        """
        super().__init__(
            dataframe,
            similarity_matrix,
            reset_different_topic_entries_flag,
            df_name,
            ProbabilityMatrixPath
        )
        if os.path.isfile(self.ProbabilityMatrixPath):
            self.probability_matrix = torch.load(self.ProbabilityMatrixPath)
        else:
            self._ResetSameBiasEntries(bias_by_topic=True)
            if reset_different_topic_entries_flag:
                self._ResetDifferentTopicEntries()
            self.probability_matrix = self._CalcProbabilities()
            reset_str = ["no_reset", "reset"]
            torch.save(self.probability_matrix, f"CE_prob_matrix_{reset_str[reset_different_topic_entries_flag]}_different_topic_entries_flag_{df_name}")

    def _CalcProbabilities(self):
        """
            The formula for each similarity entry X:
            X_TEMP =(1-(X/sum(X_row_entries)) - low cross entropy means more similar so we want to flip the scores:
            the lower the cross entropy score - the higher the prob will be
            new_row_entry = X_TEMP/sum(X_temp_row_entries)
            return: new_row_entry
        """
        zeroed_rows_indexes = self._GetProbabilityMatrixZerosRows()
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
        probability_matrix = self._PutZerosInRows(zeroed_rows_indexes, probability_matrix)
        return probability_matrix
