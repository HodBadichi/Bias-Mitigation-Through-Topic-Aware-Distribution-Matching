import re
import logging
import os
import string

import gensim
from gensim import corpora
import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline


def IsAscii(word_str):
    """
    Check whether the string contains non-ascii character
    :param word_str:string
    :return: True if the string is ONLY ascii , false otherwise
    """
    return all(ord(c) < 128 for c in word_str)


def CleanDocument(dirty_document):
    """
    Process 'dirty' document for better LDA preformance
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
    Since 'train_split' uses the same seed - the train and test are always the same

    :return: tuple of dataframes , train and test
    """
    # Used for pre-processing  requires internet connection
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

    data_directory = os.path.join(os.pardir, 'data')
    os.makedirs(data_directory, exist_ok=True)

    full_data_path = os.path.join(os.pardir, 'data', 'abstract_2005_2020_full.csv')
    clean_data_path = os.path.join(os.pardir, 'data', 'abstract_2005_2020_full_clean_LDA.csv')

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


def GetAllModelTopicsWords(model):
    x = model.show_topics(model.num_topics - 1)
    all_words = []
    for topic, word in x:
        all_words.append(re.sub('[^A-Za-z ]+', '', word).split())

    flat_list = [word for sub_word in all_words for word in sub_word]
    return flat_list


def LoadLDAModel(filepath):
    """
    Function which wraps Gensim`s load API
    :param filepath:file path where the saved model is located
    :return: the loaded model
    """
    return gensim.models.ldamodel.LdaModel.load(filepath)


def ShowEvaluationGraphs(file_path, smooth=False, poly_deg=None):
    """
    Print on the screen metrics results against number of topics
    :param file_path:CSV file which holds in a single column 'Number of topics' and different measures
    :param smooth: Boolean flag, if True it smooth the results graph
    :param poly_deg: Int, matches a polynomial of degree 'poly_deg'  to the results graph
    :return:None
    """
    figure, axis = plt.subplots(2, 3)
    figure.set_size_inches(18.5, 10.5)
    train_df = pd.read_csv(file_path)
    column_names = train_df.columns

    for measure, ax in zip(column_names, axis.ravel()):
        if measure == 'Topics':
            continue
        train_scores = train_df[measure].tolist()

        X_Y_Spline = make_interp_spline(train_df.Topics.tolist(), train_scores)
        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(train_df.Topics.min(), train_df.Topics.max(), 500)
        Y_ = X_Y_Spline(X_)
        if poly_deg is not None:
            coefs = np.polyfit(train_df.Topics.tolist(), train_scores, poly_deg)
            y_poly = np.polyval(coefs, train_df.Topics.tolist())
            ax.plot(train_df.Topics.tolist(), train_scores, "o", label="data points")
            ax.plot(train_df.Topics, y_poly, label="Validation", color='red')
        elif smooth is False:
            ax.plot(train_df.Topics, train_scores, label="Validation", color='red')
        else:
            ax.plot(X_, Y_, label="Validation", color='red')
        ax.set_title(measure + " Measure ")
        ax.set_xlabel("number of topics")
        ax.set_ylabel("measure values")
        ax.legend()
    plt.show()


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