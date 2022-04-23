from abc import abstractmethod
import torch
import pandas as pd


class SimilarityMatrix:
    def __init__(self, documents_dataframe):
        """
            :param documents_dataframe
            :param bias_by_topic True= bias threshold is the topic`s woman_rate mean, False =bias threshold is 0.5
            :param similarity_type : "cross_entropy" = topic probs cross entropy, "cosine_similarity" = bert doc embeddings cosine_similarity
        """
        self.documents_dataframe = documents_dataframe
        self.matrix = None

    @abstractmethod
    def _calc_similarities(self):
        pass

