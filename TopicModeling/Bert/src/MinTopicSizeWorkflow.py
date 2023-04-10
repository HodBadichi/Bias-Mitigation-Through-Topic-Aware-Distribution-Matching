import os
import sys
if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from bertopic import BERTopic

from TopicModeling.Utils.TopcModelingUtils import getCurrRunTime
from TopicModeling.Bert.src.BertUtils import PrepareData, BertCoherenceGraph, BertCoherenceEvaluate, BertVisualize
from TopicModeling.Bert.src.hparams_config import hparams

"""
Workflow for tuning bertTopic model over the `min_topic_size` parameter
"""


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
        model = BERTopic(verbose=True, n_gram_range=n_gram_range, min_topic_size=topic_size, calculate_probabilities=True)
        _, _ = model.fit_transform(docs)
        current_model_path = rf'n_gram_{n_gram_range[0]}_{n_gram_range[1]}_nr_topics_{topic_size}_{getCurrRunTime()}'
        model.save(os.path.join(saved_models_directory, current_model_path))
        saved_models_list.append(current_model_path)
    return saved_models_list


def RunMinTopicSizeWorkflow():
    np.random.seed(42)
    saved_models_directory_path = os.path.join(os.pardir, 'saved_models')
    results_directory_path = os.path.join(os.pardir, 'results')
    os.makedirs(saved_models_directory_path, exist_ok=True)
    os.makedirs(results_directory_path, exist_ok=True)

    train_set, test_set = PrepareData()

    trained_models_list = RunTuningProcess(
        train_set=train_set,
        saved_models_directory=saved_models_directory_path,
        min_topic_size_range=hparams['min_topic_size_range'],
        n_gram_range=hparams['n_gram_range'],
    )

    # trained_models_list = ['n_gram_1_1_nr_topics_395_05_06_2022_193226', 'n_gram_1_1_nr_topics_400_05_06_2022_193226']


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
    # to run on Technion lambda server use: sbatch -c 2 --gres=gpu:1 run_on_server.sh -o run.out
    RunMinTopicSizeWorkflow()
