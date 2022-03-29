from pprint import pprint
import numpy as np
import pandas as pd
import contractions
import string
import nltk
from Metrics import LDAMetrics
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split
nltk.download('stopwords',quiet=True)
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet',quiet=True)
nltk.download('omw-1.4',quiet=True)
import logging
import csv
np.random.seed(42)
import matplotlib.pyplot as plt

def is_ascii(s) -> bool:
    return all(ord(c) < 128 for c in s)

def clean_document(dirty_document):
    stop_words = stopwords.words('english')
    stop_words.extend(['whether'])
    dirty_document=str(dirty_document)
    # convert to lowercase
    dirty_document = ' '.join([word.lower() for word in dirty_document.split()])
    # remove punctuation
    dirty_document=''.join([letter if letter not in string.punctuation else " " for letter in dirty_document ])
    # expand contractions
    dirty_document = ' '.join([contractions.fix(word) for word in dirty_document.split()])
    # remove numbers
    dirty_document=' '.join([word for word in dirty_document.split() if word[0].isalpha() and is_ascii(word)])
    # lemmatization
    dirty_document=' '.join([WordNetLemmatizer().lemmatize(word) for word in dirty_document.split()])
    # remove short words
    dirty_document = ' '.join([word.strip() for word in dirty_document.split() if len(word.strip()) >= 3])
    #remove stopWords
    clean_document=' '.join([word for word in dirty_document.split() if word not in stop_words])
    return clean_document

def clean_text(documents_df):
    return documents_df.apply(clean_document)

def show_eval_graphs():
    validation_file_path=r'C:\Users\katac\PycharmProjects\LDAmodeling\results\models_v3\test_evaluation_v3.csv'
    train_file_path = r'C:\Users\katac\PycharmProjects\LDAmodeling\results\models_v5\train_evaluation_v5.csv'
    figure, axis = plt.subplots(2, 3)
    figure.set_size_inches(18.5, 10.5)
    train_df = pd.read_csv(train_file_path)
    # validation_df = pd.read_csv(validation_file_path)
    column_names = train_df.columns

    for measure,ax in zip(column_names,axis.ravel()):
        if measure =='Topics':
            continue
        train_scores = train_df[measure].tolist()
        # validation_scores=validation_df[measure].tolist()
        ax.plot(train_df.Topics.tolist(),train_scores ,label="Train")
        # ax.plot(validation_df.Topics.tolist(),validation_scores, label="Validation")
        ax.set_title(measure+" Measure ")
        ax.set_xlabel("number of topics")
        ax.set_ylabel("measure values")
        ax.legend()
    plt.show()
    exit(0)

def load_lda_model(filepath):
    return gensim.models.ldamodel.LdaModel.load(filepath)

def extract_lda_params(data_set):
    texts_data = [str(x).split() for x in np.squeeze(data_set).values.tolist()]
    id2word = corpora.Dictionary(texts_data)
    id2word.filter_extremes(no_below=10, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in texts_data]
    return texts_data,corpus,id2word
4
if (__name__ == '__main__'):
    show_eval_graphs()
    exit(0)
    #init logger
    # logging.basicConfig(filename='gensim.log',
    #                     format="%(asctime)s:%(levelname)s:%(message)s",
    #                     level=logging.INFO)


    #Clean data and save it
    # ds = pd.read_csv(r'abstract_2005_2020_full.csv', encoding='utf8')
    # documents_df = ds[["title_and_abstract"]]
    # new_series= clean_text(documents_df["title_and_abstract"])
    # new_series.to_csv("clean_abs.csv", index=False)


    #Load clean Data
    train_data_path = r'C:\Users\katac\PycharmProjects\LDAmodeling\Data\clean_lda_train.csv'
    training_set = pd.read_csv(train_data_path,encoding='utf8')

    test_data_path=r'C:\Users\katac\PycharmProjects\LDAmodeling\Data\clean_lda_test.csv'
    test_set = pd.read_csv(test_data_path,encoding='utf8')

    #Gensim LDA prepration - create corpus and id2word
    train_texts,train_corpus,train_id2word=extract_lda_params(training_set)
    test_texts,test_corpus,text_id2word=extract_lda_params((test_set))

    #Number of Topics Tuning
    topics_range =range(1,101,5)
    #Keys are meant to write CSV headers,values are dummy values
    my_dict = {"Topics": 6, "u_mass": 5, "c_uci": 4, "c_npmi": 3, "c_v": 2, "perplexity": 1}

    for num_of_topics in topics_range:
        print(f"Number of topics currently is {num_of_topics}")
        curr_lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                                         id2word=train_id2word,
                                                         num_topics=num_of_topics,
                                                         random_state=42,
                                                         update_every=1,
                                                         chunksize=300,
                                                         passes=1,
                                                         iterations = 50
                                                        )
        curr_lda_model.save(fr'C:\Users\katac\PycharmProjects\LDAmodeling\results\models_v5\topics_{num_of_topics}')


        #Get results on train set
        train_results_path=r'C:\Users\katac\PycharmProjects\LDAmodeling\results\models_v5\train_evaluation_v5.csv'
        with open(train_results_path, "a") as csv_file:
            my_metrics = LDAMetrics(curr_lda_model,train_corpus,train_texts)
            writer = csv.DictWriter(csv_file, my_dict.keys())
            result_dict = my_metrics.evaluate_all_metrics()
            result_dict['Topics'] = num_of_topics
            writer.writerow(result_dict)


        #Get results on test set
        test_results_path =r'C:\Users\katac\PycharmProjects\LDAmodeling\results\models_v5\test_evaluation_v5.csv'
        with open(test_results_path,"a") as csv_file:
            my_metrics = LDAMetrics(curr_lda_model,test_corpus,test_texts)
            writer = csv.DictWriter(csv_file, my_dict.keys())
            result_dict = my_metrics.evaluate_all_metrics()
            result_dict['Topics'] = num_of_topics
            writer.writerow(result_dict)






















