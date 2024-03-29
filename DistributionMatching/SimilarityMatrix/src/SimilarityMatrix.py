from abc import abstractmethod

"""
SimilarityMatrix implementation, an abstract which holds some kind of similarity metric between all the documents
in a dataframe. where the (i,j) entrance represents the similarity between document number 'i' and number 'j'
"""


class SimilarityMatrix:
    def __init__(self, documents_dataframe, df_name, ):
        """
        :param documents_dataframe: dataframe, all the documents
        :param df_name: String, whether it is train_dataset or test_dataset
        :return:None
        """
        self.documents_dataframe = documents_dataframe
        self.df_name = df_name
        self.matrix = None

    @abstractmethod
    def _CalcSimilarities(self):
        """
        this function calculate all the similarities between the documents in the dataframe
        """
        pass
