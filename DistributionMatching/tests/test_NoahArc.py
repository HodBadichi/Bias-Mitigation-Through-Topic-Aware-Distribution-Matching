import os
import pandas as pd
import torch

from DistributionMatching.NoahArc.NoahArcFactory import NoahArcFactory
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory


def test_no_nans():
    dataframe_path = os.path.join(os.pardir, os.pardir, 'data', 'abstract_2005_2020_gender_and_topic.csv')
    dataframe = pd.read_csv(dataframe_path, encoding='utf8')
    similarity_matrix = SimilarityMatrixFactory.create(dataframe, similarity_metric="cross_entropy")
    noah_arc = NoahArcFactory.create(similarity_metric="cross_entropy",
                                     reset_different_topic_entries_flag=True,
                                     similarity_matrix=similarity_matrix)
    NoahArcFactory.save(noah_arc, "../NoahArc/valid_matrix")
    assert torch.isnan(noah_arc.probability_matrix).any() == False