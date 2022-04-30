import sys
sys.path.append('/home/mor.filo/nlp_project/')
from DistributionMatching.SimilarityMatrix.SimilarityMatrix import SimilarityMatrix
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, BertForMaskedLM
from DistributionMatching.utils import config
from DistributionMatching.text_utils import clean_abstracts, TextUtils, break_sentence_batch
import torch
# from TopicModeling.Bert.src.PubMed_BertTopic import bert_apply_clean
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class SimilarityMatrixCS(SimilarityMatrix):
    def __init__(self, documents_dataframe, SimilarityMatrixPath):
        super().__init__(documents_dataframe, SimilarityMatrixPath)
        if(os.path.isfile(self.SimilarityMatrixPath)):
            self.matrix = torch.load(self.SimilarityMatrixPath)
        else:
            self.matrix = self._calc_similarities()


    def _calc_similarities(self):
        '''
            denote ce(i,j) : the cosine similarity of doc i embeddings and doc j embeddings
            create the cosine similarity similarity matrix where each value
            similarity_matrix[i][j] = (i embeddings)dot(j embeddings)/max(l2_norm(i embeddings)*l2_norm(j embeddings),eps)
        '''
        # if 'clean_title_and_abstract' not in self.documents_dataframe.columns:
        #     clean_abstracts = bert_apply_clean(self.documents_dataframe["title_and_abstract"])
        # else:
        clean_abstracts = self.documents_dataframe['broken_abstracts']
        SentenceTransformerModel = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_embeddings = SentenceTransformerModel.encode(clean_abstracts,convert_to_tensor=True)
        self.matrix = torch.as_tensor(cosine_similarity(sentence_embeddings, sentence_embeddings))
        torch.save(self.matrix, "sim_matrix_with_bert_clean")


