import os
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from DistributionMatching.SimilarityMatrix.src.SimilarityMatrix import SimilarityMatrix

"""SimilarityMatrixCS implementation
inherits from 'SimilarityMatrix' , measure similarity between documents with cosine similarity  
"""


class SimilarityMatrixCS(SimilarityMatrix):
    def __init__(self, documents_dataframe, df_name):
        super().__init__(documents_dataframe, df_name)
        matrix_file_name = f"CS_sim_matrix_{df_name}"
        similarity_matrix_path = os.path.join(os.pardir,os.pardir,os.pardir, 'data', matrix_file_name)

        if os.path.isfile(similarity_matrix_path):
            print("CS similarity matrix already exists, Loading ...")
            self.matrix = torch.load(similarity_matrix_path)
        else:
            print("Calculates CS similarity matrix...")
            self.matrix = self._CalcSimilarities()
            print("Saving CS similarity matrix...")
            torch.save(self.matrix, similarity_matrix_path)

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
