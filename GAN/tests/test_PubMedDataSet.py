import os
import sys

if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

import pandas as pd

from GAN.GANPubMed.src.PubMedDataSet import PubMedDataSet
from GAN.Utils.src.TextUtils import CleanAbstracts
from GAN.Utils.src import Utils as GAN_utils


def test_getitem():
    dataframe_path = os.path.join(
        os.pardir,
        os.pardir,
        os.pardir,
        'data',
        'abstract_2005_2020_gender_and_topic.csv'
    )
    if not os.path.exists(dataframe_path):
        GAN_utils.GenerateGANdataframe()
    dataframe = pd.read_csv(dataframe_path, encoding='utf8')
    dataframe = CleanAbstracts(dataframe)
    pubmed_dataset = PubMedDataSet(dataframe)
    # no matching doc PMID: 26476466
    no_matching_doc_index = pubmed_dataset.documents_dataframe.index[
        pubmed_dataset.documents_dataframe['PMID'] == 26476466].item()
    assert pubmed_dataset.documents_dataframe['PMID'][no_matching_doc_index] == 26476466
    item = pubmed_dataset.__getitem__(no_matching_doc_index)
    assert item['origin_document'] == no_matching_doc_index
    assert item['unbiased'] == dataframe['broken_abstracts'][no_matching_doc_index]
    assert item['biased'] is None

    # matching doc PMID: 16230722 - biased
    matching_doc_index = pubmed_dataset.documents_dataframe.index[
        pubmed_dataset.documents_dataframe['PMID'] == 16230722].item()
    assert pubmed_dataset.documents_dataframe['PMID'][matching_doc_index] == 16230722
    item = pubmed_dataset.__getitem__(matching_doc_index)
    assert item['origin_document'] == matching_doc_index
    assert item['unbiased'] is not None
    assert item['biased'] == dataframe['broken_abstracts'][matching_doc_index]
