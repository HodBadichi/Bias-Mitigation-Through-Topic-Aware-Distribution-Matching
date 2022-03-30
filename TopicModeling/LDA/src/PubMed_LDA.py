import numpy as np
import pandas as pd
import string
import nltk
import gensim
import logging
import re
import csv
import tqdm
import os
import matplotlib.pyplot as plt
from TopicModeling.Utils.Metrics import LDAMetrics
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric

import time
# Used for pre-processing  requires internet connection
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)



def is_ascii(str) -> bool:
    """
    Checks whether the string contains non-ascii character
    :param str:string
    :return: True if the string is ONLY ascii , false otherwise
    """
    return all(ord(c) < 128 for c in str)


def clean_document(dirty_document):
    """
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
    dirty_document = ' '.join([word for word in dirty_document.split() if word[0].isalpha() and is_ascii(word)])
    # lemmatization
    dirty_document = ' '.join([WordNetLemmatizer().lemmatize(word) for word in dirty_document.split()])
    # remove short words
    dirty_document = ' '.join([word.strip() for word in dirty_document.split() if len(word.strip()) >= 3])
    # remove stopWords
    clean_document = ' '.join([word for word in dirty_document.split() if word not in stop_words])
    return clean_document


def clean_text(documents_df):
    """

    :param documents_df: Panda Series , each entry is a document
    :return: Panda Series , each entry is a cleaned document after pre-processing
    """
    return documents_df.apply(clean_document)






def show_evaluation_graphs(file_path):
    """
    Print on the screen metrics results against number of topics
    :param file_path:CSV file which holds in a single column 'Number of topics' and different measures
    :return: None
    """
    figure, axis = plt.subplots(2, 3)
    figure.set_size_inches(18.5, 10.5)
    train_df = pd.read_csv(file_path)
    column_names = train_df.columns

    for measure, ax in zip(column_names, axis.ravel()):
        if measure == 'Topics':
            continue
        train_scores = train_df[measure].tolist()
        ax.plot(train_df.Topics.tolist(), train_scores, label="Train")
        ax.set_title(measure + " Measure ")
        ax.set_xlabel("number of topics")
        ax.set_ylabel("measure values")
        ax.legend()
    plt.show()
    exit(0)


def load_lda_model(filepath):
    """
    Function which wraps Gensim`s load API
    :param filepath:file path where the saved model is located
    :return: the loaded model
    """
    return gensim.models.ldamodel.LdaModel.load(filepath)


def extract_lda_params(data_set):
    """
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
    return texts_data, corpus, id2word


4
if __name__ == '__main__':
    np.random.seed(42)
    # filepath = r'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\logs\lda_model_60_topics.log'
    # gensim_log_convergence_parser(filepath)
    # exit(0)
    # init Gensim logger
    gensim_logger_path = fr'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\logs\lda_model_60_topics_S.log'
    logging.basicConfig(filename=gensim_logger_path,
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)

    # Clean and save data
    # saved_cleaned_data_path = r''
    # full_data_path = r''
    # full_documents_df = pd.read_csv(full_data_path, encoding='utf8')
    # title_and_abstract_df = full_documents_df[["title_and_abstract"]]
    # clean_title_and_abstract_df = clean_text(title_and_abstract_df["title_and_abstract"])
    # clean_title_and_abstract_df.to_csv(saved_cleaned_data_path, index=False)

    # Load clean data
    train_data_path = r'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\data\clean_lda_train.csv'
    training_set = pd.read_csv(train_data_path, encoding='utf8')

    # Gensim LDA preparation - create corpus and id2word
    train_texts, train_corpus, train_id2word = extract_lda_params(training_set)


    coherence_cv_logger = CoherenceMetric(corpus=train_corpus, logger='shell', coherence='c_v', texts=train_texts)
    perplexity_logger = PerplexityMetric(corpus=train_corpus, logger='shell')

    # Number of Topics Tuning
    topics_range = [61]
    # Keys are meant to write CSV headers later on ,values are dummy values
    my_dict = {"Topics": 6, "u_mass": 5, "c_uci": 4, "c_npmi": 3, "c_v": 2, "perplexity": 1}


    # List of the different iterations to try
    iterations = [50, 100, 400]

    # The number of passes to use - could change depending on requirements
    passes = 1

    # for iteration in iterations:
    #
    #     # Add text to logger to indicate new model
    #     logging.debug(f'Start of model: {iteration} iterations')
    #
    #     # Create model - note callbacks argument uses list of created callback loggers
    #     model = models.ldamodel.LdaModel(corpus=train_corpus,
    #                                      id2word=train_id2word,
    #                                      num_topics=61,
    #                                      eval_every=20,
    #                                      passes=passes,
    #                                      iterations=iteration,
    #                                      random_state=42,
    #                                      callbacks=[perplexity_logger, coherence_cv_logger])
    #
    #     # Add text to logger to indicate end of this model
    #     logging.debug(f'End of model: {iteration} iterations')
    #
    #     # Save models so they aren't lost
    #     if not os.path.exists(f"lda_{iteration}i50p/"):
    #         os.makedirs(f"lda_{iteration}i50p/")
    #
    #     model.save(f"lda_{iteration}i50p/lda_{iteration}i50p.model")

    all_metrics = pd.DataFrame()

    for iteration in tqdm.tqdm(iterations):
        model = models.ldamodel.LdaModel.load(f"lda_{iteration}i50p/lda_{iteration}i50p.model")
        df = pd.DataFrame.from_dict(model.metrics)

        df['docs_converged'] = find_doc_convergence(5, iteration, r'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\logs\lda_model_60_topics_S.log')
        df['iterations'] = iteration
        df['topics'] = 5

        df = df.reset_index().rename(columns={'index': 'pass_num'})

        all_metrics = pd.concat([all_metrics, df])

    for metric in ['Coherence', 'Perplexity','docs_converged']:

        fig, axs = plt.subplots(1, 1, figsize=(20, 7))

        # Each plot to show results for all models with the same topic number
        for i, topic_number in enumerate([5]):
            filtered_topics = all_metrics[all_metrics['topics'] == topic_number]
            for label, df in filtered_topics.groupby(['iterations']):
                print(label)
                df.plot(x='pass_num', y=metric, ax=axs, label=label)

            axs.set_xlabel(f"Pass number")
            axs.legend()
            axs.set_ylim([all_metrics[metric].min() * 0.9, all_metrics[metric].max() * 1.1])

        if metric == 'docs_converged':
            fig.suptitle('Documents converged', fontsize=20)
        else:
            fig.suptitle(metric, fontsize=20)
    exit(0)

    for num_of_topics in topics_range:
        print("topic run")
        print(train_id2word)
        print(num_of_topics)
        passes=1
        curr_lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                                         id2word=train_id2word,
                                                         num_topics=num_of_topics,
                                                         random_state=42,
                                                         update_every=1,
                                                         chunksize=300,
                                                         passes=passes,
                                                         iterations=iterations,
                                                         callbacks=[perplexity_logger,coherence_cv_logger])
        logging.debug(f'End of model: {iterations} iterations')

        # saved_model_path = r''
        # curr_lda_model.save(saved_model_path)

        # Save results of train set
        # train_results_path = r''
        # with open(train_results_path, "a") as csv_file:
        #     # Initialize 'LDAMetrics' class
        #     my_metrics = LDAMetrics(curr_lda_model, train_corpus, train_texts)
        #     writer = csv.DictWriter(csv_file, my_dict.keys())
        #     result_dict = my_metrics.evaluate_all_metrics()
        #     result_dict['Topics'] = num_of_topics
        #     writer.writerow(result_dict)
