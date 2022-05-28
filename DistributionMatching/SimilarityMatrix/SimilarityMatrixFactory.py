import sys
sys.path.append('/home/mor.filo/nlp_project/')
from DistributionMatching.SimilarityMatrix.SimilarityMatrixCE import SimilarityMatrixCE
from DistributionMatching.SimilarityMatrix.SimilarityMatrixCS import SimilarityMatrixCS
import pandas as pd

class SimilarityMatrixFactory:
    @staticmethod
    def create(documents_dataframe, similarity_metric, df_name, SimilarityMatrixPath = ''):
        if similarity_metric == "cross_entropy":
            return SimilarityMatrixCE(documents_dataframe, df_name, SimilarityMatrixPath)
        elif similarity_metric == "cosine_similarity":
            return SimilarityMatrixCS(documents_dataframe, df_name, SimilarityMatrixPath)
        raise NotImplementedError("`SimilarityMatrixFactory` unsupported metric")



if __name__ == '__main__':
    df = pd.read_csv(rf'../../data/abstract_2005_2020_gender_and_topic.csv',
                     encoding='utf8')
    train_df = df.loc[df['belongs_to_group'] == 'train']
    train_df = train_df.reset_index()
    matrix = SimilarityMatrixFactory.create(train_df, similarity_metric='cosine_similarity', df_name="train")
