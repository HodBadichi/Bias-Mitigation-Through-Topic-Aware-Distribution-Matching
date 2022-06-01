import logging
import os
import string

from gensim import corpora
import gdown
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


from TopicModeling.Utils.TopcModelingUtils import IsAscii


def CleanDocument(dirty_document):
    """
    Process 'dirty' document for better LDA performance
    :param dirty_document: string of the dirty document
    :return: string of the clean document
    """
    stop_words = stopwords.words('english')
    stop_words.extend(['whether'])
    dirty_document = str(dirty_document)
    # convert to lowercase
    dirty_document = ' '.join([word.lower() for word in dirty_document.split()])
    # remove punctuation
    dirty_document = ''.join([letter if letter not in string.punctuation else " " for letter in dirty_document])
    # remove numbers
    dirty_document = ' '.join([word for word in dirty_document.split() if word[0].isalpha() and IsAscii(word)])
    # lemmatization
    dirty_document = ' '.join([WordNetLemmatizer().lemmatize(word) for word in dirty_document.split()])
    # remove short words
    dirty_document = ' '.join([word.strip() for word in dirty_document.split() if len(word.strip()) >= 3])
    # remove stopWords
    clean_document = ' '.join([word for word in dirty_document.split() if word not in stop_words])
    return clean_document


def CleanText(documents_df):
    """
    :param documents_df: Panda Series , each entry is a document
    :return: Panda Series , each entry is a cleaned document after pre-processing
    """
    return documents_df.apply(CleanDocument)


def PrepareData():
    """
    Load and process the data,
    Since 'train_split' uses the same seed - the train_dataset and test_dataset are always the same

    :return: tuple of dataframes , train_dataset and test_dataset
    """
    # Used for pre-processing  requires internet connection
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

    data_directory = os.path.join(os.pardir, os.pardir, os.pardir, 'data')
    os.makedirs(data_directory, exist_ok=True)

    full_data_path = os.path.join(os.pardir, os.pardir, os.pardir, 'data', 'abstract_2005_2020_full.csv')
    clean_data_path = os.path.join(os.pardir, os.pardir, os.pardir, 'data', 'abstract_2005_2020_full_clean_LDA.csv')

    #   Load full dataframe
    if not os.path.exists(full_data_path):
        logging.info("Downloading dataframe ...")
        url = 'https://drive.google.com/uc?id=1xmifhCZ4IljgjUEY73QLVPnk99Y-A04A'
        gdown.download(url, full_data_path, quiet=False)
    else:
        logging.info("Dataframe already exists...")

    #  Load clean dataframe
    if not os.path.exists(clean_data_path):
        logging.info("Cleaning full dataframe ...")
        full_documents_df = pd.read_csv(full_data_path, encoding='utf8')
        title_and_abstract_df = full_documents_df[["title_and_abstract"]]
        clean_title_and_abstract_df = CleanText(title_and_abstract_df["title_and_abstract"])
        clean_title_and_abstract_df.to_csv(clean_data_path, index=False)
    else:
        logging.info("Clean dataframe already exists...")

    clean_dataframe = pd.read_csv(clean_data_path, encoding='utf8')
    training_dataset, test_dataset = train_test_split(clean_dataframe, train_size=0.9, random_state=42)
    return training_dataset, test_dataset


def GetLDAParams(data_set):
    """
    Convert dataframe to gensim`s LDA parameters

    :param data_set: pandas dataframe object , each entry is a document
    :return:
          texts_data: list of lists where each inner-list represent a single document : [[dog,cat,mouse],[..],[..]]
          corpus:Gensim corpus parameter for creating the LDA model
          id2word:Gensim dictionary parameter for creating the LDA model
    """
    texts_data = [str(x).split() for x in np.squeeze(data_set).values.tolist()]
    id2word = corpora.Dictionary(texts_data)
    # filter words which appear in less than 10 documents , or in more than 50% of the documents
    id2word.filter_extremes(no_below=10, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in texts_data]
    return {'texts': texts_data, 'corpus': corpus, 'id2word': id2word}
