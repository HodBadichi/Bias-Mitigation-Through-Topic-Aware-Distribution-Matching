import pyLDAvis.gensim_models
from gensim import corpora
import gensim.models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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



train_data_path = r'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\data\hod_clean_lda_train.csv'
training_set = pd.read_csv(train_data_path, encoding='utf8')
train_texts, train_corpus, train_id2word = extract_lda_params(training_set)
lda_model = load_lda_model(r'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\results\model_16')



#Coloured charts
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in train_texts for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(4, 4, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.CSS4_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 100000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
plt.show()