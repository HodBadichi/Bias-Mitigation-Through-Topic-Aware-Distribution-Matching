from torch.utils.data import Dataset
from DistributionMatching.NoahArc.NoahArcFactory import NoahArcFactory
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory
import DistributionMatching.utils as project_utils
from DistributionMatching.utils import config
from DistributionMatching.text_utils import TextUtils


class PubMedDataSet(Dataset):
    def __init__(self, documents_dataframe):
        self.Matcher = PubMedDataSet._build_noah_arc(documents_dataframe,config['similarity_metric'])
        self.modified_document_df = self.Matcher.documents_dataframe
        self.tu = TextUtils()

    def __len__(self):
        # defines len of epoch
        return len(self.modified_document_df)

    def create_text_field(self, sent_list):
        sent_list_filtered_by_words = [' '.join(self.tu.word_tokenize(sent)) for sent in sent_list]
        return '<BREAK>'.join(sent_list_filtered_by_words)


    def __getitem__(self, index):
        result = {}
        document_df = self.modified_document_df
        similar_doc_index = self.Matcher.get_match(index)[0]
        index_text = self.create_text_field(document_df.iloc[index]["sentences"])
        similar_doc_index_text = self.create_text_field(document_df.iloc[similar_doc_index]["sentences"])
        if project_utils.are_women_minority(index, self.Matcher.documents_dataframe):
            result['biased'] = (index_text,
                                document_df.iloc[index]["female_rate"],
                                document_df.iloc[index]["major_topic"])

            result['unbiased'] = (similar_doc_index_text,
                                  document_df.iloc[similar_doc_index]["female_rate"],
                                  document_df.iloc[similar_doc_index]["major_topic"])
        else:
            result['biased'] = (similar_doc_index_text,
                                document_df.iloc[similar_doc_index]["female_rate"],
                                document_df.iloc[similar_doc_index]["major_topic"])

            result['unbiased'] = (index_text,
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
