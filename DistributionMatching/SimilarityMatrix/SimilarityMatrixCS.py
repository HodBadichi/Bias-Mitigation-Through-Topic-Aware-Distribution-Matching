from DistributionMatching.SimilarityMatrix.SimilarityMatrix import SimilarityMatrix
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityMatrixCS(SimilarityMatrix):
    def __init__(self, documents_dataframe):
        super().__init__(documents_dataframe)
        self.matrix = self._calc_similarities()

    def _calc_similarities(self):
        '''
            denote ce(i,j) : the cosine similarity of doc i embeddings and doc j embeddings
            create the cosine similarity similarity matrix where each value
            similarity_matrix[i][j] = (i embeddings)dot(j embeddings)/max(l2_norm(i embeddings)*l2_norm(j embeddings),eps)
        '''
        pass
