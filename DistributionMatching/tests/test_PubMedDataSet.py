import pandas as pd

from DistributionMatching.PubMed.PubMedDataSet import PubMedDataSet
from DistributionMatching.utils import config


def test_getitem():
    dataframe = pd.read_csv(config['data']['gender_and_topic_path'],encoding='utf8')
    pubmed_dataset = PubMedDataSet(dataframe)
    #no matching doc PMID: 26476466
    no_matching_doc_index = pubmed_dataset.documents_dataframe.index[pubmed_dataset.documents_dataframe['PMID'] == 26476466]
    assert pubmed_dataset['PMID'][no_matching_doc_index] == 26476466
    item = pubmed_dataset.__getitem__(no_matching_doc_index)
    assert item['origin_document'] == no_matching_doc_index
    assert item['unbiased'] == pubmed_dataset.create_text_field(pubmed_dataset['sentences'][no_matching_doc_index])
    assert item['biased'] is None

    #matching doc PMID: 16230722 - biased
    matching_doc_index = pubmed_dataset.documents_dataframe.index[pubmed_dataset.documents_dataframe['PMID'] == 16230722]
    assert pubmed_dataset['PMID'][matching_doc_index] == 16230722
    item = pubmed_dataset.__getitem__(matching_doc_index)
    assert item['origin_document'] == matching_doc_index
    assert item['unbiased'] is not None
    assert item['biased'] == pubmed_dataset.create_text_field(pubmed_dataset['sentences'][matching_doc_index])




