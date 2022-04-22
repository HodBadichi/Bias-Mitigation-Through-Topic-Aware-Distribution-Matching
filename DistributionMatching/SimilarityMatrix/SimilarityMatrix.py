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
        try:
            self.documents_dataframe = pd.read_csv("SimMatrixDf", encoding='utf8')
            self.matrix = torch.load("SimMatrixTensor")
        except FileNotFoundError:
            print("Couldn`t find Similarity matrix on disk , Calculating...")
            self.documents_dataframe = documents_dataframe
            self.matrix = None

    @abstractmethod
    def _calc_similarities(self):
        pass

    def save(self):
        torch.save(self.matrix, "SimMatrixTensor")
        self.documents_dataframe.to_csv( "SimMatrixDf")
