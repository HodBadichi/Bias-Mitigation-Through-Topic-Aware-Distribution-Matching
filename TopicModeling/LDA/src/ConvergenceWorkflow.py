from gensim import models
import logging
from TopicModeling.LDA.src.LDAUtils import GetLDAParams
from TopicModeling.LDA.src.LDAUtils import PrepareData
import os
import matplotlib.pyplot as plt


def InitLogger():
    """
    Intialize a global logger to parse when displaying the graph
    :return: None
    """
    file_name = 'model_callbacks_passes.txt'
    LoggerPath = os.path.join(os.pardir, 'logs', file_name)
    logging.basicConfig(
        filename=LoggerPath,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.NOTSET)


def PlotConvergenceGraph():
    """
    Plot the convergence graph of the trained model using the logger file.
    :return:None
    """
    passes = []
    coherence = []
    #   Close logger
    logging.shutdown()

    #   Parse logger
    file_name = 'model_callbacks_passes.txt'
    LoggerPath = os.path.join(os.pardir, 'logs', file_name)
    for line in open(LoggerPath):
        if 'Start' in line:
            short = " ".join(line.split()[:-1])
            passes.append(short.split()[-1])
        if 'coherence' in line:
            coherence.append(line.split()[-1])

    passes = [int(i) for i in passes]
    coherence = [round(float(i), 3) for i in coherence]
    print(passes)
    print(coherence)
    plt.plot(passes, coherence)
    plt.xlabel("passes")
    plt.ylabel("c_v coherence value")
    plt.show()


def RunConvergenceProcess(
        train_LDA_params,
        number_of_topics,
        passes_range=range(1, 5),
        iterations=250,
        chunksize=300,
):
    """

    :param train_LDA_params: dictionary of {'corpus','texts','id2word'}
    :param number_of_topics: hyperparam
    :param passes_range: passes range to exhibit convergence
    :param iterations: hyperparam
    :param chunksize: hyperparam
    :return:
    """
    params_dictionary = {
        'passes_range': passes_range,
        'iterations': iterations,
        'chunksize': chunksize,
        'number_of_topics': number_of_topics,
    }
    logging.info(params_dictionary)

    for i in passes_range:
        # Add text to logger to indicate new model
        logging.debug(f'Start of model: {i} passes')

        # Create model - note callbacks argument uses list of created callback loggers
        curr_lda_model = models.ldamodel.LdaModel(
            corpus=train_LDA_params['corpus'],
            id2word=train_LDA_params['id2word'],
            num_topics=number_of_topics,
            update_every=1,
            chunksize=chunksize,
            passes=i,
            iterations=iterations,
            random_state=42
        )

        coherencemodel = models.CoherenceModel(
            model=curr_lda_model, texts=train_LDA_params['texts'], corpus=train_LDA_params['corpus'],
            dictionary=train_LDA_params['id2word'], coherence='c_v')
        logging.debug(f"c_v coherence : {coherencemodel.get_coherence()}")
        # Add text to logger to indicate end of this model
        logging.debug(f'End of model: {i} passes')
        PlotConvergenceGraph()


def RunConvergenceExperiment():
    number_of_topics = 1
    InitLogger()
    train_set, _ = PrepareData()
    train_LDA_params = GetLDAParams(train_set)
    RunConvergenceProcess(train_LDA_params, number_of_topics)


if __name__ == '__main__':
    RunConvergenceExperiment()
