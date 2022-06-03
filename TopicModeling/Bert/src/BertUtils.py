import os
import csv

from bertopic import BERTopic
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import gdown

from TopicModeling.Utils.Metrics import BertTopicMetrics
from TopicModeling.Utils.TopcModelingUtils import getCurrRunTime, IsAscii


def BertVisualize(models_dir, result_dir, model_name, top_n_topics, n_words_per_topic):
    """
    Create html with 2 figures:
    bar chart (main words in the topic)
     https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.visualize_barchart
    visualize topics (2d representation of the topic space)
     https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.visualize_topics
    :param models_dir:Path , to models dir
    :param result_dir: Path ,results directory path
    :param model_name: string, model name
    :param top_n_topics: int, n topics to visualize
    :param n_words_per_topic: int, n words per topic to visualize
    :return: None
    """
    # visualize model results and save it to html file
    model_path = os.path.join(models_dir, model_name)
    loaded_model = BERTopic.load(model_path)
    fig1 = loaded_model.visualize_barchart(top_n_topics=top_n_topics, n_words=n_words_per_topic)
    fig2 = loaded_model.visualize_topics()
    with open(os.path.join(result_dir, model_name + '.html'), 'a') as f:
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))


def BertCoherenceGraph(evaluation_file_name, result_dir):
    """
    Visualize the experiment coherence graph of all trained models
    :param evaluation_file_name:Path, evaluation csv file
    :param result_dir:Path, results dir
    :return:
    """
    evaluation_path = os.path.join(result_dir, evaluation_file_name)
    figure, axis = plt.subplots(2, 3)
    figure.set_size_inches(18.5, 10.5)
    train_df = pd.read_csv(evaluation_path)
    column_names = train_df.columns

    for measure, ax in zip(column_names, axis.ravel()):
        if (measure == "Topics") or (measure == "Model"):
            continue
        train_scores = train_df[measure].tolist()
        # validation_scores=validation_df[measure].tolist()
        ax.plot(train_df.Topics.tolist(), train_scores, label="Train")
        # ax.plot(validation_df.Topics.tolist(),validation_scores, label="Validation")
        ax.set_title(measure + " Measure ")
        ax.set_xlabel("Number of topics")
        ax.set_ylabel("Measure values")
        ax.legend()
    plt.savefig(rf'{result_dir}\coherence_graphs_{getCurrRunTime()}.pdf')
    plt.close()


def CleanDocument(dirty_document):
    """
    Process 'dirty' document for better Bert performance
    :param dirty_document: string of the dirty document
    :return: string of the clean document
    """
    # remove numbers
    dirty_document = ' '.join([word for word in dirty_document.split() if word[0].isalpha() and IsAscii(word)])
    # remove short words
    clean_document = ' '.join([word.strip() for word in dirty_document.split() if len(word.strip()) >= 3])
    return clean_document


def CleanText(documents_df):
    return documents_df.apply(CleanDocument)


def BertCoherenceEvaluate(train_set, models_dir, models_list, results_dir):
    """
    Calculate the coherence of the model using our "BertTopicMetrics" module that is based on gensim`s coherence
    :param train_set: dataframe
    :param models_dir: path,saved models directory
    :param models_list: list of strings,containing models to evaluate
    :param results_dir: path, save the results of each model
    :return: None
    """
    # calculate coherence
    result_path = os.path.join(results_dir, f'evaluate_{getCurrRunTime()}.csv')
    my_dict = {"Model": "Model", "Topics": "Topics", "u_mass": "u_mass", "c_uci": "c_uci", "c_npmi": "c_npmi",
               "c_v": "c_v"}
    docs = train_set.title_and_abstract.dropna().to_list()
    with open(result_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, my_dict.keys())
        writer.writerow(my_dict)
        for model in models_list:
            loaded_model = BERTopic.load(os.path.join(models_dir,model))
            loaded_topics = loaded_model._map_predictions(loaded_model.hdbscan_model.labels_)
            my_metrics = BertTopicMetrics(loaded_model, docs, loaded_topics)
            result_dict = my_metrics.EvaluateAllMetrics()
            result_dict['Model'] = model
            result_dict['Topics'] = len(loaded_model.get_topics())
            writer.writerow(result_dict)


def PrepareData():
    """
    Load and process the data,
    Since 'train_split' uses the same seed - the train_dataset and test_dataset are always the same

    :return: tuple of dataframes , train_dataset and test_dataset
    """
    data_directory = os.path.join(os.pardir, os.pardir, os.pardir, 'data')
    os.makedirs(data_directory, exist_ok=True)

    full_data_path = os.path.join(os.pardir, os.pardir, os.pardir, 'data', 'abstract_2005_2020_full.csv')
    clean_data_path = os.path.join(os.pardir, os.pardir, os.pardir, 'data', 'abstract_2005_2020_full_clean_BERT.csv')

    #   Load full dataframe
    if not os.path.exists(full_data_path):
        print("Downloading dataframe ...")
        url = 'https://drive.google.com/uc?id=1xmifhCZ4IljgjUEY73QLVPnk99Y-A04A'
        gdown.download(url, full_data_path, quiet=False)
    else:
        print("Dataframe already exists...")

    #  Load clean dataframe
    if not os.path.exists(clean_data_path):
        print("Cleaning full dataframe ...")
        full_documents_df = pd.read_csv(full_data_path, encoding='utf8')
        title_and_abstract_df = full_documents_df[["title_and_abstract"]]
        clean_title_and_abstract_df = CleanText(title_and_abstract_df["title_and_abstract"])
        clean_title_and_abstract_df.to_csv(clean_data_path, index=False)
    else:
        print("Clean dataframe already exists...")

    clean_dataframe = pd.read_csv(clean_data_path, encoding='utf8')
    training_dataset, test_dataset = train_test_split(clean_dataframe, train_size=0.9, random_state=42)
    return training_dataset, test_dataset


def BertShowTopicFrequency(model_path, model_name):
    """
    Create a bar chart of frequency of topics (major topic) among all documents
    :param model_path:BertTopic model path to visualize
    :param model_name: BertTopic model name to visualize
    :return: None
    """
    loaded_model = BERTopic.load(model_path)
    frequency = loaded_model.get_topic_freq()
    topic_freq = {}
    for index, row in frequency.iterrows():
        if row[0] != -1:
            topic_freq[row[0]] = row[1]
        else:
            count_general_topic = row[1]
    plt.bar(list(topic_freq.keys()), topic_freq.values(), color='g')
    plt.title(f"{model_name}\nTopic -1 with {count_general_topic} documents")
    plt.xlabel("Topic Number")
    plt.ylabel("Frequency among documents")
    plt.show()
