import os
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from DistributionMatching.SimilarityMatrix.src.SimilarityMatrix import SimilarityMatrix

"""SimilarityMatrixCS implementation
inherits from 'SimilarityMatrix' , measure similarity between documents with cosine similarity  
"""


class SimilarityMatrixCS(SimilarityMatrix):
    def __init__(self, documents_dataframe, df_name, SimilarityMatrixPath):
        super().__init__(documents_dataframe, df_name, SimilarityMatrixPath)
        if os.path.isfile(self.SimilarityMatrixPath):
            self.matrix = torch.load(self.SimilarityMatrixPath)
        else:
            self.matrix = self._CalcSimilarities()
            torch.save(self.matrix, f"CS_sim_matrix_{df_name}")

    def _CalcSimilarities(self):
        """
            denote ce(i,j) : the cosine similarity of doc i embeddings and doc j embeddings
            Create the cosine similarity similarity matrix where each value
            similarity_matrix[i][j] = (i embeddings)dot(j embeddings)/max(l2_norm(i embeddings)*l2_norm(j embeddings),eps)
        """
        clean_abstracts = self.documents_dataframe['broken_abstracts']
        SentenceTransformerModel = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_embeddings = SentenceTransformerModel.encode(clean_abstracts, convert_to_tensor=True)
        matrix = torch.as_tensor(cosine_similarity(sentence_embeddings.cpu(), sentence_embeddings.cpu()))
        return matrix
