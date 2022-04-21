from DistributionMatching.SimilartyMatrix import SimilartyMatrix
import torch
import pandas as pd
import numpy as np
import DistributionMatching.utils as project_utils
import os


class NoahArc:
    def __init__(self, SimilartyMatrix=None):
        try:
            self.documents_dataframe=pd.read_csv("NoahArcDf", encoding='utf8')
            self.probability_matrix=torch.load("NoahArcProbMatrixTensor")
            self.similarty_matrix = None
        except FileNotFoundError:
            self.similarty_matrix = SimilartyMatrix.matrix
            self.documents_dataframe=SimilartyMatrix.documents_df
            self.probability_matrix = self._CalcProbabilities()

    def _CalcProbabilities(self):
        """
                The formula for each similarity entry X:
                X_TEMP =(1-(X/sum(X_row_entries)) - low cross entropy means more similar so we want to flip the scores:
                the lower the cross entropy score - the higher the prob will be
                new_row_entry = X_TEMP/sum(X_temp_row_entries)
                return: new_row_entry
        """
        # Transform float32 Tensor to float64 Tensor to prevent numeric errors (rounding to 0 in small values)
        prob_matrix = self.similarty_matrix.double()
        rows_sum = torch.sum(prob_matrix, 1).reshape((-1,1))
        prob_matrix = torch.div(prob_matrix, rows_sum)
        # No need to flip (1-x) zero and one values
        prob_matrix = torch.where((prob_matrix != 0) & (prob_matrix != 1), 1-prob_matrix, prob_matrix)
        rows_sum = torch.sum(prob_matrix, 1).reshape(-1,1)
        prob_matrix = torch.div(prob_matrix, rows_sum)
        return prob_matrix

    def GetMatch(self, document_index):
        """
        :param document_index: The document index we need to find a matching document for
        :return: matching document index,matching document PMID to use with other dfs
        """
        probabilities = self.probability_matrix[document_index]
        similar_doc_index = np.random.choice(range(0, len(self.probability_matrix)), 1, p=probabilities)
        return similar_doc_index[0], self.documents_dataframe['PMID'][similar_doc_index]

    @staticmethod
    def IsNoahSaved():
        return os.path.exists("NoahArcProbMatrixTensor") and os.path.exists("NoahArcDf")

    def Save(self):
        self.documents_dataframe.to_csv( "NoahArcDf")
        torch.save(self.probability_matrix,"NoahArcProbMatrixTensor")

if __name__ == '__main__':
    df = pd.read_csv(project_utils.DATAFRAME_PATH,
                     encoding='utf8')
    matrix = SimilartyMatrix(df)
    # matrix.ResetDiffTopicEntries()
    # matrix.ResetSameBiasEntries()
    # matrix.DropUnWantedDocsAndReset()
    # matrix.SaveMatrix()
    Prob = NoahArc(matrix)
    print("loaded to memory")
    print(Prob.GetMatch(42))
    print(f"42 PMID is {Prob.document_df['PMID'][42]}")
    # Prob.SaveMatrix()

