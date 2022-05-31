from DistributionMatching.SimilarityMatrix.src.SimilarityMatrixCE import SimilarityMatrixCE
from DistributionMatching.SimilarityMatrix.src.SimilarityMatrixCS import SimilarityMatrixCS

""" SimilarityMatrixFactory implementation factory of 'SimilarityMatrix'
"""


class SimilarityMatrixFactory:
    @staticmethod
    def create(documents_dataframe, similarity_metric, df_name, SimilarityMatrixPath=''):
        if similarity_metric == "cross_entropy":
            return SimilarityMatrixCE(documents_dataframe, df_name, SimilarityMatrixPath)
        elif similarity_metric == "cosine_similarity":
            return SimilarityMatrixCS(documents_dataframe, df_name, SimilarityMatrixPath)
        raise NotImplementedError("`SimilarityMatrixFactory` unsupported metric")
