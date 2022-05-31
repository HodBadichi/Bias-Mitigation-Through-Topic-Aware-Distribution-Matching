import sys
sys.path.append(rf'C:\Users\morfi\PycharmProjects\NLP_project')

import pandas as pd
import os
from bertopic import BERTopic
from TopicModeling.Utils.Metrics import BertTopicMetrics
import csv
import matplotlib.pyplot as plt
import dateutil.parser as parser

def is_ascii(s) -> bool:
    return all(ord(c) < 128 for c in s)


def bert_clean_document(dirty_document):
    # # lemmatization
    # dirty_document=' '.join([WordNetLemmatizer().lemmatize(word) for word in dirty_document.split()])
    # remove numbers
    dirty_document = ' '.join([word for word in dirty_document.split() if word[0].isalpha() and is_ascii(word)])
    # remove short words
    clean_document = ' '.join([word.strip() for word in dirty_document.split() if len(word.strip()) >= 3])
    return clean_document


def bert_apply_clean(documents_df):
    return documents_df.apply(bert_clean_document)


def bert_preprocess(train_data_path, save_path):
    """
    Clean data for bert and save it
    """
    for data in ["train", "test"]:
        documents_df = pd.read_csv(train_data_path, encoding='utf8')
        new_series = bert_apply_clean(documents_df["title_and_abstract"])
        # new_series = new_series.dropna()
        # new_series = documents_df[documents_df['title_and_abstract'].notna()]
        new_series.to_csv(save_path, index=False)


def bert_train(train_data_path, model_path, min_topic_size, n_gram_range=(1, 1)):
    """
    input: documents path, min_topic_size (limitation to the ‘HDBSCAN’ algorithem)
    and n_gram_range (1,1 for unigram, 1,2 for bigram ...)
    """
    # Load clean Data
    documents_df = pd.read_csv(train_data_path, encoding='utf8')
    # convert to list
    docs = documents_df.title_and_abstract.to_list()
    for topic_size in min_topic_size:
        model = BERTopic(verbose=True, n_gram_range=n_gram_range, min_topic_size=topic_size)
        topics, probabilities = model.fit_transform(docs)
        model.save(rf'{model_path}_n_gram_{n_gram_range[0]}_{n_gram_range[1]}_nr_topics_{topic_size}_new')


def bert_coherence_evaluate(train_data_path, models_dir, models_list, result_path):
    """
    calculate the coherence of the model using our "BertTopicMetrics" module that is based on gensim coherent evaluate
    """
    # calculate coherence
    my_dict = {"Model": 6, "Topics": 5, "u_mass": 4, "c_uci": 3, "c_npmi": 2, "c_v": 1}
    documents_df = pd.read_csv(train_data_path, encoding='utf8')
    docs = documents_df.title_and_abstract.to_list()
    for model in models_list:
        loaded_model = BERTopic.load(rf"{models_dir}\{model}")
        loaded_topics = loaded_model._map_predictions(loaded_model.hdbscan_model.labels_)
        with open(result_path, "a") as csv_file:
            my_metrics = BertTopicMetrics(loaded_model, docs, loaded_topics)
            result_dict = my_metrics.EvaluateAllMetrics()
            writer = csv.DictWriter(csv_file, my_dict.keys())
            result_dict['Model'] = model
            result_dict['Topics'] = len(loaded_model.get_topics())
            writer.writerow(result_dict)


def bert_coherence_graph(evaluation_path, result_dir):
    """
    input: path to the coherence data of the models we are interested in, calculated by "bert_coherence_evaluate"
    output: coherence to number of topic graph for each coherence metric (cv, u_mass...)
    """
    figure, axis = plt.subplots(2, 3)
    figure.set_size_inches(18.5, 10.5)
    train_df = pd.read_csv(evaluation_path)
    column_names = train_df.columns

    for measure, ax in zip(column_names, axis.ravel()):
        if measure == 'Topics':
            continue
        train_scores = train_df[measure].tolist()
        # validation_scores=validation_df[measure].tolist()
        ax.plot(train_df.Topics.tolist(), train_scores, label="Train")
        # ax.plot(validation_df.Topics.tolist(),validation_scores, label="Validation")
        ax.set_title(measure + " Measure ")
        ax.set_xlabel("number of topics")
        ax.set_ylabel("measure values")
        ax.legend()
    plt.savefig(rf'{result_dir}\coherence_graphs.pdf')
    plt.close()


def bert_visualize(model_path, result_dir, model_name, top_n_topics, n_words_per_topic):
    """
    create html with 2 figures:
    bar chart (main words in the topic) https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.visualize_barchart
    visualize topics (2d representation of the topic space) https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.visualize_topics
    """
    # visualize model results and save it to html file
    loaded_model = BERTopic.load(model_path)
    fig1 = loaded_model.visualize_barchart(top_n_topics=top_n_topics, n_words=n_words_per_topic)
    fig2 = loaded_model.visualize_topics()
    with open(rf'{result_dir}\{model_name}.html', 'a') as f:
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))


def bert_topic_over_time(model_path, result_dir, model_name, documents_path):
    """
    create the "topic over time" visualization from the documents "title and abstract" and year of publication (from the "date" filed)
    (https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.visualize_topics_over_time)
    """
    loaded_model = BERTopic.load(model_path)
    loaded_topics = loaded_model._map_predictions(loaded_model.hdbscan_model.labels_)
    documents_df = pd.read_csv(documents_path,encoding='utf8')
    docs = documents_df.title_and_abstract.to_list()
    timestamps = documents_df.date.to_list()
    timestamps = [parser.parse(item).year for item in timestamps]
    topics_over_time = loaded_model.topics_over_time(docs, loaded_topics, timestamps)
    fig = loaded_model.visualize_topics_over_time(topics_over_time, topics=[0,1,2,3,4,5,6,7,8,9])
    fig.write_html(rf'{result_dir}\{model_name}_topics_over_time.html')

def bert_show_topic_frequency(model_path, model_name):
    """
    create a bar chart of frequency of topics (major topic) among all documents
    """
    loaded_model = BERTopic.load(model_path)
    frequency = loaded_model.get_topic_freq()
    topic_freq = {}
    for index, row in frequency.iterrows():
        if (row[0] != -1):
            topic_freq[row[0]]=row[1]
        else:
            count_general_topic = row[1]
    plt.bar(list(topic_freq.keys()), topic_freq.values(), color='g')
    plt.title(f"{model_name}\nTopic -1 with {count_general_topic} documents")
    plt.xlabel("Topic Number")
    plt.ylabel("Frequency among documents")
    plt.show()

if (__name__ == '__main__'):

    documents_path = rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\data\abstract_2005_2020_gender.csv'
    clean_documents_path = rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\data\clean_abstract_2005_2020_gender.csv'
    bert_preprocess(documents_path, clean_documents_path)


    bert_train(train_data_path=rf'clean_bert_train.csv',
               model_path=rf'bertTopic_train(81876)',
               min_topic_size=[330,331,332,333,334], n_gram_range=(1, 1)
               )


    bert_coherence_evaluate(
        train_data_path=rf'clean_bert_train.csv',
        models_dir=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src',
        models_list = [rf"bertTopic_train(81876)_n_gram_1_1_nr_topics_333"],
        result_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\results\bert\train_evaluation_n_gram_1_1.csv'
        )

    bert_topic_over_time(model_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src\bertTopic_train(81876)_n_gram_1_1_min_topic_size_50',
                        result_dir=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src',
                        model_name="bertTopic_train(81876)_n_gram_1_1_min_topic_size_50",
                        documents_path=rf'clean_bert_train.csv'
                        )

    models = ["bertTopic_train(81876)_n_gram_1_1_nr_topics_15","bertTopic_train(81876)_n_gram_1_1_nr_topics_16"],
    for model in models:
        bert_visualize(
            model_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src\{model}',
            result_dir=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src',
            model_name=model,
            top_n_topics=20,
            n_words_per_topic=15
            )

    bert_coherence_graph(evaluation_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src\train_evaluation.csv',
                         result_dir=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src',
                         )

    bert_show_topic_frequency(model_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\TopicModeling\Bert\src\bertTopic_train(81876)_n_gram_1_2_min_topic_size_300',
                              model_name='bertTopic_train(81876)_n_gram_1_2_min_topic_size_300'
                              )