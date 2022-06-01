import sys
sys.path.append('/home/mor.filo/nlp_project/')
from torch.utils.data import Dataset
from DistributionMatching.NoahArc.src.NoahArcFactory import NoahArcFactory
from DistributionMatching.SimilarityMatrix.src.SimilarityMatrixFactory import SimilarityMatrixFactory
import DistributionMatching.Utils.Utils as project_utils
from GAN.Utils.TextUtils import TextUtils


class PubMedDataSet(Dataset):
    def __init__(self, documents_dataframe, hparams, df_name, SimilarityMatrixPath, ProbabilityMatrixPath):
        self.hparams = hparams
        self.documents_dataframe = documents_dataframe
        self.Matcher = self.build_noah_arc(df_name, SimilarityMatrixPath, ProbabilityMatrixPath)
        self.tu = TextUtils()

    def __len__(self):
        # defines len of epoch
        return len(self.documents_dataframe)

    def __getitem__(self, index):
        batch_entry = {}
        similar_document_index = self.Matcher.GetMatch(index)
        if similar_document_index is not None:
            similar_document_broken_abstracts = self.documents_dataframe['broken_abstracts'][similar_document_index]
        else:
            similar_document_broken_abstracts = ""
        origin_document_broken_abstracts = self.documents_dataframe['broken_abstracts'][index]

        #This field will ease MLM loss calculation
        batch_entry['origin_text'] = origin_document_broken_abstracts

        if project_utils.AreWomenMinority(index, self.Matcher.documents_dataframe):
            batch_entry['biased'] = origin_document_broken_abstracts
            batch_entry['unbiased'] = similar_document_broken_abstracts
        else:
            batch_entry['biased'] = similar_document_broken_abstracts
            batch_entry['unbiased'] = origin_document_broken_abstracts
        return batch_entry

    def build_noah_arc(self, df_name, SimilarityMatrixPath, ProbabilityMatrixPath):
        similarity_matrix = SimilarityMatrixFactory.Create(self.documents_dataframe, self.hparams.similarity_metric,
                                                           df_name, SimilarityMatrixPath)
        probability_matrix = NoahArcFactory.Create(self.documents_dataframe, self.hparams.similarity_metric,
                                                   similarity_matrix, self.hparams.reset_different_topic_entries_flag,
                                                   df_name, ProbabilityMatrixPath)
        return probability_matrix
