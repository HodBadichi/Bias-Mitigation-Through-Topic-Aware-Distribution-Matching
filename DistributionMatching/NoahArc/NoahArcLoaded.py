from abc import ABC

from DistributionMatching.NoahArc.NoahArc import NoahArc
import torch


class NoahArcLoaded(NoahArc, ABC):
    def __init__(self, probability_matrix, documents_dataframe):
        self.probability_matrix = probability_matrix
        self.documents_dataframe = documents_dataframe
