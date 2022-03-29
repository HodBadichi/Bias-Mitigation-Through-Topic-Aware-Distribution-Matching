import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from Metrics import BertTopicMetrics
import csv


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


def bert_preprocess(documents_df):
    # Clean data for bert and save it
    for data in ["train", "test"]:
        documents_df = pd.read_csv(rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\Data\{data}.csv',
                                   encoding='utf8')
        new_series = bert_apply_clean(documents_df["title_and_abstract"])
        # new_series = new_series.dropna()
        # new_series = documents_df[documents_df['title_and_abstract'].notna()]
        new_series.to_csv(rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\Data\clean_bert_{data}.csv',
                          index=False)


def bert_train(train_data_path, model_path, min_topic_size_range=[10], n_gram_range=(1, 1)):
    # Load clean Data
    documents_df = pd.read_csv(train_data_path, encoding='utf8')
    # convert to list
    docs = documents_df.title_and_abstract.to_list()
    for topic_size in min_topic_size_range:
        model = BERTopic(verbose=True, n_gram_range=n_gram_range, min_topic_size=topic_size)
        topics, probabilities = model.fit_transform(docs)
        model.save(rf'{model_path}_n_gram_{n_gram_range[0]}_{n_gram_range[1]}_min_topic_size_{topic_size}')


def bert_coherence_evaluate(train_data_path, models_dir, models_list, result_path):
    # calculate coherence
    my_dict = {"Model": 5, "u_mass": 4, "c_uci": 3, "c_npmi": 2, "c_v": 1}
    documents_df = pd.read_csv(train_data_path, encoding='utf8')
    docs = documents_df.title_and_abstract.to_list()
    for model in models_list:
        loaded_model = BERTopic.load(rf"{models_dir}\{model}")
        loaded_topics = loaded_model._map_predictions(loaded_model.hdbscan_model.labels_)
        with open(result_path, "a") as csv_file:
            my_metrics = BertTopicMetrics(loaded_model, docs, loaded_topics)
            result_dict = my_metrics.evaluate_all_metrics()
            writer = csv.DictWriter(csv_file, my_dict.keys())
            result_dict['Model'] = model
            writer.writerow(result_dict)


def bert_visualize(model_path, result_dir, model_name, top_n_topics, n_words_per_topic):
    # visualize model results and save it to html file
    loaded_model = BERTopic.load(model_path)
    fig1 = loaded_model.visualize_barchart(top_n_topics=top_n_topics, n_words=n_words_per_topic)
    fig2 = loaded_model.visualize_topics()
    with open(rf'{result_dir}\{model_name}.html', 'a') as f:
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))


if (__name__ == '__main__'):
    bert_train(train_data_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\Data\clean_bert_train.csv',
               model_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\results\bert\bertTopic_train(81876)',
               min_topic_size_range=[25, 50, 100, 300, 800, 1600], n_gram_range=(1, 1)
               )

    bert_coherence_evaluate(
        train_data_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\Data\clean_bert_train.csv',
        models_dir=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\results\bert',
        models_list=["bertTopic_train(81876)_n_gram_1_1_min_topic_size_25",
                     "bertTopic_train(81876)_n_gram_1_1_min_topic_size_50",
                     "bertTopic_train(81876)_n_gram_1_1_min_topic_size_100",
                     "bertTopic_train(81876)_n_gram_1_1_min_topic_size_300"],
        result_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\results\bert\train_evaluation.csv'
        )

    bert_visualize(
        model_path=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\results\bert\bertTopic_train(81876)_n_gram_1_1_min_topic_size_25',
        result_dir=rf'C:\Users\{os.getlogin()}\PycharmProjects\LDAmodeling\results\bert',
        model_name='bertTopic_6000_n_gram_1_3',
        top_n_topics=30,
        n_words_per_topic=10
        )
