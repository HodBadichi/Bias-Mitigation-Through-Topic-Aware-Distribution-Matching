import sys
sys.path.append('/home/mor.filo/nlp_project/')
import pandas as pd
import os
# import dill
# import mgzip
import torch

from DistributionMatching.NoahArc.NoahArcCE import NoahArcCE
from DistributionMatching.NoahArc.NoahArcCS import NoahArcCS
from DistributionMatching.NoahArc._NoahArcLoaded import _NoahArcLoaded
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory


class NoahArcFactory:
    @staticmethod
    def create(dataframe, similarity_metric, similarity_matrix, reset_different_topic_entries_flag, ProbabilityMatrixPath = ''):
        if similarity_metric == "cross_entropy":
            return NoahArcCE(dataframe, similarity_matrix, reset_different_topic_entries_flag, ProbabilityMatrixPath)
        elif similarity_metric == "cosine_similarity":
            return NoahArcCS(dataframe, similarity_matrix, reset_different_topic_entries_flag, ProbabilityMatrixPath)
        raise NotImplementedError("`NoahArcFactory` unsupported metric")

    # @staticmethod
    # def save(noah_arc_object, save_path):
    #     database = {'matrix': noah_arc_object.probability_matrix,
    #                 'dataframe': noah_arc_object.documents_dataframe}
    #     with mgzip.open(save_path, "wb") as file:
    #         dill.dump(database, file)
    #
    # @staticmethod
    # def load(load_path):
    #     with mgzip.open(load_path, "rb") as file:
    #         database = dill.load(file)
    #     probability_matrix = database['matrix']
    #     documents_dataframe = database['dataframe']
    #     return _NoahArcLoaded(probability_matrix, documents_dataframe)


if __name__ == '__main__':
    df = pd.read_csv(rf'../../data/abstract_2005_2020_gender_and_topic.csv', encoding='utf8')
    matrix = SimilarityMatrixFactory.create(df, similarity_metric='cross_entropy')
    # torch.save(matrix, "CE_sim_matrix")
    # matrix = torch.load("sim_matrix_with_BERTopic_clean")
    # print(f'sim_matrix_with_BERTopic_clean: {matrix.shape}')
    # print(f'sim_matrix_with_BERTopic_clean: {type(matrix)}')
    # print(f'sim_matrix_with_BERTopic_clean: {matrix[1][0]}')
    Prob = NoahArcFactory.create(df, 'cross_entropy', matrix, False)
    torch.save(Prob.probability_matrix, "CE_prob_matrix_no_reset_different_topic_entries_flag")
    assert not torch.isnan(Prob.probability_matrix).any()

