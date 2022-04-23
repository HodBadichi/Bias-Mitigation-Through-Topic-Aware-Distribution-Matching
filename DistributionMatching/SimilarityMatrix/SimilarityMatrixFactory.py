from DistributionMatching.SimilarityMatrix.SimilarityMatrixCE import SimilarityMatrixCE
from DistributionMatching.SimilarityMatrix.SimilarityMatrixCS import SimilarityMatrixCS
import DistributionMatching.utils as project_utils
import pandas as pd
import torch
import os


class SimilarityMatrixFactory:
    @staticmethod
    def create(documents_dataframe, similarity_metric):
        if similarity_metric == "cross_entropy":
            return SimilarityMatrixCE(documents_dataframe)
        elif similarity_metric == "cosine_similarity":
            return SimilarityMatrixCS(documents_dataframe)
        raise NotImplementedError("`SimilarityMatrixFactory` unsupported metric")


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.pardir, os.pardir, "data", "abstract_2005_2020_gender_and_topic.csv"),
                     encoding='utf8')
    matrix = SimilarityMatrixFactory.create(df, similarity_metric='cross_entropy')
