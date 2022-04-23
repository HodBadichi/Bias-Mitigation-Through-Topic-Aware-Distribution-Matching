from torch.utils.data import Dataset

from DistributionMatching.NoahArc.NoahArcFactory import NoahArcFactory
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory
import DistributionMatching.utils as project_utils
from DistributionMatching.utils import config


class PubMedDataSet(Dataset):
    def __init__(self, documents_dataframe):
        self.Matcher = PubMedDataSet._build_noah_arc(documents_dataframe,config['similarity_metric'])
        self.modified_document_df = self.Matcher.documents_dataframe

    def __len__(self):
        # defines len of epoch
        return len(self.modified_document_df)

    def __getitem__(self, index):
        result = {}
        document_df = self.modified_document_df
        similar_doc_index = self.Matcher.get_match(index)[0]
        if project_utils.are_women_minority(index, self.Matcher.documents_dataframe):
            result['biased'] = (document_df.iloc[index]["title_and_abstract"],
                                document_df.iloc[index]["female_rate"],
                                document_df.iloc[index]["major_topic"])

            result['unbiased'] = (document_df.iloc[similar_doc_index]["title_and_abstract"],
                                  document_df.iloc[similar_doc_index]["female_rate"],
                                  document_df.iloc[similar_doc_index]["major_topic"])
        else:
            result['biased'] = (document_df.iloc[similar_doc_index]["title_and_abstract"],
                                document_df.iloc[similar_doc_index]["female_rate"],
                                document_df.iloc[similar_doc_index]["major_topic"])

            result['unbiased'] = (document_df.iloc[index]["title_and_abstract"],
                                  document_df.iloc[index]["female_rate"],
                                  document_df.iloc[index]["major_topic"])
        return result

    @staticmethod
    def _build_noah_arc(dataframe, similarity_metric):
        target_file = f"noaharc_{similarity_metric}"
        try:
            NoahArcFactory.load(target_file)
        except FileNotFoundError:
            similarity_matrix = SimilarityMatrixFactory.create(dataframe, similarity_metric)
            probability_matrix = NoahArcFactory.create(similarity_metric,
                                                       config['reset_different_topic_entries_flag'],
                                                       similarity_matrix)
            NoahArcFactory.save(probability_matrix, target_file)
        return probability_matrix
