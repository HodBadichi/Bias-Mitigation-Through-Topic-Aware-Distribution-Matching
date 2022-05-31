import pandas as pd

from GAN.PubMed.PubMedDataSet import PubMedDataSet
from DistributionMatching.utils import config
from GAN.PubMed.text_utils import clean_abstracts


def test_getitem():
    dataframe = pd.read_csv(config['data']['gender_and_topic_path'], encoding='utf8')
    dataframe = clean_abstracts(dataframe)
    pubmed_dataset = PubMedDataSet(dataframe)
    #no matching doc PMID: 26476466
    no_matching_doc_index = pubmed_dataset.documents_dataframe.index[pubmed_dataset.documents_dataframe['PMID'] == 26476466].item()
    assert pubmed_dataset.documents_dataframe['PMID'][no_matching_doc_index] == 26476466
    item = pubmed_dataset.__getitem__(no_matching_doc_index)
    assert item['origin_document'] == no_matching_doc_index
    assert item['unbiased'] == dataframe['broken_abstracts'][no_matching_doc_index]
    assert item['biased'] is None

    #matching doc PMID: 16230722 - biased
    matching_doc_index = pubmed_dataset.documents_dataframe.index[pubmed_dataset.documents_dataframe['PMID'] == 16230722].item()
    assert pubmed_dataset.documents_dataframe['PMID'][matching_doc_index] == 16230722
    item = pubmed_dataset.__getitem__(matching_doc_index)
    assert item['origin_document'] == matching_doc_index
    assert item['unbiased'] is not None
    assert item['biased'] == dataframe['broken_abstracts'][matching_doc_index]
