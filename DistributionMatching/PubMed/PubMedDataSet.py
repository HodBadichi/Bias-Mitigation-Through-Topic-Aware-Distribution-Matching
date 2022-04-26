from torch.utils.data import Dataset
from DistributionMatching.NoahArc.NoahArcFactory import NoahArcFactory
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory
import DistributionMatching.utils as project_utils
from DistributionMatching.utils import config
from DistributionMatching.text_utils import TextUtils


class PubMedDataSet(Dataset):
    def __init__(self, documents_dataframe):
        self.Matcher = PubMedDataSet._build_noah_arc(documents_dataframe, config['similarity_metric'])
        self.documents_dataframe = documents_dataframe
        self.tu = TextUtils()

    def __len__(self):
        # defines length of a single epoch
        return len(self.documents_dataframe)

    def create_text_field(self, document_index):
        return self.documents_dataframe.iloc[document_index]["title_and_abstract"]
        # document_as_sentences = self.documents_dataframe.iloc[document_index]["sentences"]
        # document_filtered_by_words = [' '.join(self.tu.word_tokenize(sentence)) for sentence in document_as_sentences]
        # return '<BREAK>'.join(document_filtered_by_words)

    def __getitem__(self, index):
        batch_entry = {'origin_document': index}
        similar_document_index = self.Matcher.get_match(index)

        if similar_document_index is not None:
            similar_document_modified_text = self.create_text_field(similar_document_index)
        else:
            similar_document_modified_text = None
        origin_document_modified_text = self.create_text_field(index)

        if project_utils.are_women_minority(index, self.Matcher.documents_dataframe):
            batch_entry['biased'] = origin_document_modified_text
            batch_entry['unbiased'] = similar_document_modified_text
        else:
            batch_entry['biased'] = similar_document_modified_text
            batch_entry['unbiased'] = origin_document_modified_text
        return batch_entry

    @staticmethod
    def _build_noah_arc(dataframe, similarity_metric):
        target_file = f"noaharc_{similarity_metric}"
        try:
            return NoahArcFactory.load(target_file)
        except FileNotFoundError:
            similarity_matrix = SimilarityMatrixFactory.create(dataframe, similarity_metric)
            probability_matrix = NoahArcFactory.create(similarity_metric,
                                                       config['reset_different_topic_entries_flag'],
                                                       similarity_matrix)
            NoahArcFactory.save(probability_matrix, target_file)
            return probability_matrix
