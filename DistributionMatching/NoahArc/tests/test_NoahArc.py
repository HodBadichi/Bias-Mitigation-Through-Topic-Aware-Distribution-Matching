import os
import pandas as pd
import torch

from DistributionMatching.NoahArc.src.NoahArcFactory import NoahArcFactory
from DistributionMatching.SimilarityMatrix.src.SimilarityMatrixFactory import SimilarityMatrixFactory


def test_no_nans():
    dataframe_path = os.path.join(os.pardir, os.pardir, os.pardir, 'data', 'abstract_2005_2020_gender_and_topic.csv')
    dataframe = pd.read_csv(dataframe_path, encoding='utf8')
    similarity_matrix = SimilarityMatrixFactory.Create(dataframe, df_name='train_dataset', similarity_metric="cross_entropy")
    noah_arc = NoahArcFactory.Create(
        dataframe=dataframe,
        similarity_metric="cross_entropy",
        reset_different_topic_entries_flag=True,
        similarity_matrix=similarity_matrix,
        df_name='train_dataset'
    )
    assert torch.isnan(noah_arc.probability_matrix).any() == False


if __name__ == '__main__':
    test_no_nans()