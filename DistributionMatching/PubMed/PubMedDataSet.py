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
        # defines len of epoch
        return len(self.documents_dataframe)

    def __getitem__(self, index):
        batch_entry = {'origin_document': index}
        similar_document_index = self.Matcher.get_match(index)

        if similar_document_index is not None:
            similar_document_broken_abstracts = self.documents_dataframe['broken_abstracts'][similar_document_index]
        else:
            similar_document_broken_abstracts = None
        origin_document_broken_abstracts = self.documents_dataframe['broken_abstracts'][index]

        #This field will ease MLM loss calculation
        batch_entry['origin_text'] = origin_document_broken_abstracts

        if project_utils.are_women_minority(index, self.Matcher.documents_dataframe):
            batch_entry['biased'] = origin_document_broken_abstracts
            batch_entry['unbiased'] = similar_document_broken_abstracts
        else:
            batch_entry['biased'] = similar_document_broken_abstracts
            batch_entry['unbiased'] = origin_document_broken_abstracts
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
