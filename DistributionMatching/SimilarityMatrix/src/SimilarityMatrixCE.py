import json
import os
import torch

from DistributionMatching.SimilarityMatrix.src.SimilarityMatrix import SimilarityMatrix

"""SimilarityMatrixCE implementation
inherits from 'SimilarityMatrix' , measure similarity between documents with cross entropy  
"""


class SimilarityMatrixCE(SimilarityMatrix):
    def __init__(self, documents_dataframe, df_name, SimilarityMatrixPath):
        super().__init__(documents_dataframe, df_name, SimilarityMatrixPath)
        if os.path.isfile(self.SimilarityMatrixPath):
            self.matrix = torch.load(self.SimilarityMatrixPath)
        else:
            self.matrix = self._CalcSimilarities()
            torch.save(self.matrix, f"CE_sim_matrix_{df_name}")

    def _CalcSimilarities(self):
        probs = self.documents_dataframe['probs'].apply(lambda x: json.loads(x))
        tensor_probs = torch.as_tensor(probs)  # shape num_of_docs X num_of_topics (distribution for every doc)
        res = self._CrossEntropy(tensor_probs, tensor_probs)  # tensor shape num_of_docs X num_of_docs
        return res

    @staticmethod
    def _CrossEntropy(prob1_vector, prob2_vector, eps=1e-21):
        """
            :param prob1_vector: tensor that represent probability
            :param prob2_vector: tensor that represent probability
            :param eps: to avoid log(0)
            :return:cross entropy between prob1 and prob2 by the formula: -sum(prob1*log(prob2))
        """
        prob2_vector = torch.add(prob2_vector, eps)
        prob2_vector = torch.log(prob2_vector)
        return -prob1_vector @ prob2_vector.T
