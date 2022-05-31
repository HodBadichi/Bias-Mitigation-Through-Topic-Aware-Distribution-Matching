import torch
import os

from DistributionMatching.NoahArc.src.NoahArc import NoahArc

"""NoahArcCS implementation
inherits from 'NoahArc' , based on 'Cosine similarity' metric  
"""


class NoahArcCS(NoahArc):
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
            df_name, ProbabilityMatrixPath
        )
        if os.path.isfile(self.ProbabilityMatrixPath):
            self.probability_matrix = torch.load(self.ProbabilityMatrixPath)
        else:
            self._ResetSameBiasEntries(bias_by_topic=False)
            self.probability_matrix = self._CalcProbabilities()
            torch.save(self.probability_matrix, f"CS_prob_matrix_with_BERTopic_clean_{df_name}")

    def _CalcProbabilities(self):
        """
                The formula for each similarity entry X:
                new_row_entry = <document1 probabilities,document2 probabilities> / (|document1 probabilities |*|document2 probabilities|)
                return: new_row_entry
        """
        probability_matrix = self._similarity_matrix.double()
        probability_matrix = torch.where(
            probability_matrix < 0.0,
            torch.tensor(0, dtype=probability_matrix.dtype),
            probability_matrix
        )
        rows_sum_vector = torch.sum(probability_matrix, 1).reshape((-1, 1))
        probability_matrix = torch.div(probability_matrix, rows_sum_vector)
        return probability_matrix