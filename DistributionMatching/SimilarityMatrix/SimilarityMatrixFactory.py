from DistributionMatching.SimilarityMatrix.SimilarityMatrixCE import SimilarityMatrixCE
from DistributionMatching.SimilarityMatrix.SimilarityMatrixCS import SimilarityMatrixCS
import DistributionMatching.utils as project_utils
import pandas as pd
import torch


class SimilarityMatrixFactory:
    @staticmethod
    def create(documents_dataframe, similarity_metric):
        if similarity_metric == "cross_entropy":
            return SimilarityMatrixCE(documents_dataframe)
        elif similarity_metric == "cosine_similarity":
            return SimilarityMatrixCS(documents_dataframe)
        raise NotImplementedError("`SimilarityMatrixFactory` unsupported metric")


if __name__ == '__main__':
    df = pd.read_csv(project_utils.DATAFRAME_PATH,
                     encoding='utf8')
    matrix = SimilarityMatrixFactory.create(df, similarity_metric='cross_entropy')
    print(len(matrix.documents_dataframe))
    print(torch.count_nonzero(matrix.matrix[282]).item())
