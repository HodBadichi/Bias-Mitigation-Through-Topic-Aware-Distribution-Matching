from abc import abstractmethod


class SimilarityMatrix:
    def __init__(self, documents_dataframe, SimilarityMatrixPath):
        """
            :param documents_dataframe
        """
        self.documents_dataframe = documents_dataframe
        self.SimilarityMatrixPath = SimilarityMatrixPath
        self.matrix = None

    @abstractmethod
    def _calc_similarities(self):
        pass

