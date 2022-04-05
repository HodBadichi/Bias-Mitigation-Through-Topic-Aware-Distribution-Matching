import pandas as pd
import numpy as np
from gensim import models
import os
from gensim import corpora
### choose the callbacks classes to import
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric
from gensim.models import CoherenceModel

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


# The filename is the file that will be created with the log.
# If the file already exists, the log will continue rather than being overwritten.
from importlib import reload  # Not needed in Python 2
import logging
reload(logging)
logging.basicConfig(filename=os.path.join(os.pardir,'logs','model_callbacks_passes.log'),
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.NOTSET)

train_data_path = os.path.join(os.pardir, 'data', 'hod_clean_lda_train.csv')
training_set = pd.read_csv(train_data_path, encoding='utf8')
documents,corpus,dictionary = extract_lda_params(training_set)

# List of the different iterations to try
passes = range(60,100,20)

# The number of passes to use - could change depending on requirements

for i in passes:

    # Add text to logger to indicate new model
    logging.debug(f'Start of model: {i} passes')

    # Create model - note callbacks argument uses list of created callback loggers
    curr_lda_model = models.ldamodel.LdaModel(corpus=corpus,
             id2word=dictionary,
             num_topics=16,
             update_every=1,
            chunksize=300,
             passes=i,
             iterations=250,
            random_state=42)
    coherencemodel = models.CoherenceModel(
                model=curr_lda_model, texts=documents, corpus=corpus,
                dictionary=dictionary,coherence='c_v')
    logging.debug(f"c_v coherence : {coherencemodel.get_coherence()}")
    # Add text to logger to indicate end of this model
    logging.debug(f'End of model: {i} passes')

