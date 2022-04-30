import sys
sys.path.append('/home/mor.filo/nlp_project/')
from DistributionMatching.SimilarityMatrix.SimilarityMatrixCE import SimilarityMatrixCE
from DistributionMatching.SimilarityMatrix.SimilarityMatrixCS import SimilarityMatrixCS
import pandas as pd

class SimilarityMatrixFactory:
    @staticmethod
    def create(documents_dataframe, similarity_metric, SimilarityMatrixPath = ''):
        if similarity_metric == "cross_entropy":
            return SimilarityMatrixCE(documents_dataframe, SimilarityMatrixPath)
        elif similarity_metric == "cosine_similarity":
            return SimilarityMatrixCS(documents_dataframe, SimilarityMatrixPath)
        raise NotImplementedError("`SimilarityMatrixFactory` unsupported metric")



if __name__ == '__main__':
    df = pd.read_csv(rf'../../data/abstract_2005_2020_gender_and_topic.csv',
                     encoding='utf8')
    matrix = SimilarityMatrixFactory.create(df, similarity_metric='cosine_similarity')
