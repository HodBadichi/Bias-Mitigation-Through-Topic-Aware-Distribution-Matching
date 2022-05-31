import logging
from datetime import datetime
import csv
import os

import numpy as np
import gensim
from gensim.models import CoherenceModel

from TopicModeling.Utils.Metrics import LDAMetrics
from TopicModeling.LDA.src.LDAUtils import GetLDAParams
from TopicModeling.LDA.src.LDAUtils import PrepareData


def InitializeLogger(sFileName=None):
    """
    Intialize a global logger to document the experiment
    :param sFileName: required logger name
    :return: None
    """
    logs_directory_path = os.path.join(os.pardir, 'logs')
    os.makedirs(logs_directory_path, exist_ok=True)

    if sFileName is None:
        sFileName = f'log_{datetime.now().strftime("%d_%m_%Y_%H%M%S")}.txt'
    LoggerPath = os.path.join(os.pardir, 'logs', sFileName)
    from importlib import reload
    reload(logging)
    logging.basicConfig(
        filename=LoggerPath,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.NOTSET)


def RunTuningProcess(
        train_LDA_parameters,
        test_LDA_parameters,
        topics_range,
        passes=3,
        iterations=5,
        chunksize=300,
):
    """
    Train and Evaluate each model with a different number of topics by the topics range given from the input.
    each model is saved and evaluated by 'perplexity' and 'Coherence' measurements, when the evaluation results
    are saved as well.
    :param train_LDA_parameters: dictionary of {'corpus','texts','id2word'}
    :param test_LDA_parameters: dictionary of {'corpus','texts','id2word'}
    :param topics_range: topics range to tune
    :param passes: number of passes the model does on the whole corpus
    :param iterations: number of iterations the model does
    :param chunksize: number of documents in each iteration
    :return: None
    """
    params_dictionary = {
        'passes': passes,
        'iterations': iterations,
        'chunksize': chunksize,
        'topics_range': topics_range
    }
    logging.info(params_dictionary)

    models_directory_path = os.path.join(os.pardir, 'saved_models')
    result_directory_path = os.path.join(os.pardir, 'results')
    os.makedirs(models_directory_path, exist_ok=True)
    os.makedirs(result_directory_path, exist_ok=True)

    # Keys are meant to write CSV headers later on ,values are dummy values
    my_dict = {"Topics": 6, "u_mass": 5, "c_uci": 4, "c_npmi": 3, "c_v": 2, "perplexity": 1}
    current_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    for num_of_topics in topics_range:
        logging.info(f"Running number of topics model : {num_of_topics}")
        curr_lda_model = gensim.models.ldamodel.LdaModel(corpus=train_LDA_parameters['corpus'],
                                                         id2word=train_LDA_parameters['id2word'],
                                                         num_topics=num_of_topics,
                                                         random_state=42,
                                                         update_every=1,
                                                         chunksize=chunksize,
                                                         passes=passes,
                                                         iterations=iterations)

        saved_model_path = os.path.join(models_directory_path, f'model_{num_of_topics}_{current_time}')
        curr_lda_model.save(saved_model_path)
        # Save results of train set
        train_results_path = os.path.join(result_directory_path, fr'train_evaluation_{current_time}.csv')
        with open(train_results_path, "a") as csv_file:
            # Initialize 'LDAMetrics' class
            my_metrics = LDAMetrics(curr_lda_model, train_LDA_parameters['corpus'], train_LDA_parameters['texts'])
            writer = csv.DictWriter(csv_file, my_dict.keys())
            result_dict = my_metrics.EvaluateAllMetrics()
            result_dict['Topics'] = num_of_topics
            writer.writerow(result_dict)

        test_results_path = os.path.join(result_directory_path, fr'test_evaluation_{current_time}.csv')
        with open(test_results_path, "a") as csv_file:
            # Initialize 'LDAMetrics' class
            my_metrics = LDAMetrics(curr_lda_model, test_LDA_parameters['corpus'], test_LDA_parameters['texts'])
            writer = csv.DictWriter(csv_file, my_dict.keys())
            result_dict = my_metrics.EvaluateAllMetrics()
            result_dict['Topics'] = num_of_topics
            writer.writerow(result_dict)


def RunNumberOfTopicsExperiment():
    np.random.seed(42)

    InitializeLogger()
    train_set, test_set = PrepareData()

    #   Gensim LDA preparation - create corpus and id2word
    train_LDA_params = GetLDAParams(train_set)
    test_LDA_params = GetLDAParams(test_set)

    #   ChooseHyperparameters:
    topics_range = range(1, 10)
    RunTuningProcess(train_LDA_params, test_LDA_params, topics_range)


if __name__ == '__main__':
    RunNumberOfTopicsExperiment()
