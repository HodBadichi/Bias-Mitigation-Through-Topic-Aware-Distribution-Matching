import os
import csv

import numpy as np
import pandas as pd
import gdown
from bertopic import BERTopic
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from TopicModeling.Utils.Metrics import BertTopicMetrics
from TopicModeling.Utils.TopcModelingUtils import getCurrRunTime, IsAscii


def BertVisualize(models_dir, result_dir, model_name, top_n_topics, n_words_per_topic):
    """
    Create html with 2 figures:
    bar chart (main words in the topic) https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.visualize_barchart
    visualize topics (2d representation of the topic space) https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.visualize_topics
    :param model_path:Path ,bertTopicModel path
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
    my_dict = {"Model": "Model", "Topics": "Topics", "u_mass": "u_mass", "c_uci": "c_uci", "c_npmi": "c_npmi", "c_v": "c_v"}
    docs = train_set.title_and_abstract.dropna().to_list()
    with open(result_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, my_dict.keys())
        writer.writerow(my_dict)
        for model in models_list:
            loaded_model = BERTopic.load(rf"{models_dir}\{model}")
            loaded_topics = loaded_model._map_predictions(loaded_model.hdbscan_model.labels_)
            my_metrics = BertTopicMetrics(loaded_model, docs, loaded_topics)
            result_dict = my_metrics.EvaluateAllMetrics()
            result_dict['Model'] = model
            result_dict['Topics'] = len(loaded_model.get_topics())
            writer.writerow(result_dict)


def PrepareData():
    """
    Load and process the data,
    Since 'train_split' uses the same seed - the train and test are always the same

    :return: tuple of dataframes , train and test
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


def RunTuningProcess(train_set, saved_models_directory, min_topic_size_range, n_gram_range=(1, 1)):
    """
    Train and Evaluate each model with a different number of mmin_topics by the min_topics range given from the input.
    each model is saved and evaluated , when the evaluation results are saved as well.
    :param train_set: dataframe,training set
    :param saved_models_directory:Path , directory where models are saved
    :param min_topic_size_range:hyperparam
    :param n_gram_range:hyperparam
    :return: list of strings,of the saved models
    """
    saved_models_list = []
    # convert` to list
    docs = train_set.title_and_abstract.dropna().to_list()
    for topic_size in min_topic_size_range:
        model = BERTopic(verbose=True, n_gram_range=n_gram_range, min_topic_size=topic_size)
        topics, probabilities = model.fit_transform(docs)
        current_model_path = rf'n_gram_{n_gram_range[0]}_{n_gram_range[1]}_nr_topics_{topic_size}_{getCurrRunTime()}'
        model.save(os.path.join(saved_models_directory, current_model_path))
        saved_models_list.append(current_model_path)
    return saved_models_list


def RunMinTopicSizeExperiment():
    np.random.seed(42)
    saved_models_directory_path = os.path.join(os.pardir, os.pardir, 'saved_models')
    results_directory_path = os.path.join(os.pardir, os.pardir, 'results')
    os.makedirs(saved_models_directory_path, exist_ok=True)
    os.makedirs(results_directory_path, exist_ok=True)

    train_set, test_set = PrepareData()

    #   Choose Hyperparameters:
    min_topic_size_range = [340, 345, 350]
    n_gram_range = (1, 1)
    trained_models_list = RunTuningProcess(
        train_set=train_set,
        saved_models_directory=saved_models_directory_path,
        min_topic_size_range=min_topic_size_range,
        n_gram_range=n_gram_range,
    )
    trained_models_list = ['bertTopic_train(81876)_n_gram_1_1_min_topic_size_50_with_probs']
    BertCoherenceEvaluate(
        train_set=train_set,
        models_dir=saved_models_directory_path,
        models_list=trained_models_list,
        results_dir=results_directory_path,
    )

    for model in trained_models_list:
        BertVisualize(
            models_dir=saved_models_directory_path,
            result_dir=results_directory_path,
            model_name=model,
            top_n_topics=20,
            n_words_per_topic=15,
        )

    BertCoherenceGraph(
        evaluation_file_name=f'evaluate_{getCurrRunTime()}.csv',
        result_dir=results_directory_path,
    )


if __name__ == '__main__':
    RunMinTopicSizeExperiment()
