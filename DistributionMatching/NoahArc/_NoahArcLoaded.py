import sys
sys.path.append('/home/mor.filo/nlp_project/')
from abc import ABC

from DistributionMatching.NoahArc.NoahArc import NoahArc


class _NoahArcLoaded(NoahArc, ABC):
    """
    This class is used to serve just 'NoahFactory.load()' method,and supply a different constructor.
    """
    def __init__(self, probability_matrix, documents_dataframe):
        self.probability_matrix = probability_matrix
        self.documents_dataframe = documents_dataframe
