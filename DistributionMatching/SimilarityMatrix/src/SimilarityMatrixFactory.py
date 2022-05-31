from DistributionMatching.SimilarityMatrix.src.SimilarityMatrixCE import SimilarityMatrixCE
from DistributionMatching.SimilarityMatrix.src.SimilarityMatrixCS import SimilarityMatrixCS

""" SimilarityMatrixFactory implementation factory of 'SimilarityMatrix'
"""


class SimilarityMatrixFactory:
    @staticmethod
    def Create(documents_dataframe, similarity_metric, df_name):
        if similarity_metric == "cross_entropy":
            return SimilarityMatrixCE(documents_dataframe, df_name)
        elif similarity_metric == "cosine_similarity":
            return SimilarityMatrixCS(documents_dataframe, df_name)
        raise NotImplementedError("`SimilarityMatrixFactory` unsupported metric")
