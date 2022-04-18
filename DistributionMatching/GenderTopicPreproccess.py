import os
import pandas as pd
from bertopic import BERTopic
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

def plot_topic_female_rate_hists(data_path, loaded_model, result_path=""):
    # receives df that has "female_rate" col and "topic" col and a loaded model (to get topic name)
    documents_df = pd.read_csv(data_path, encoding='utf8')
    df1 = documents_df.groupby('major_topic')['female_rate'].apply(list).reset_index(name='female_rate')
    info_df = loaded_model.get_topic_info()
    info_df = info_df.sort_values(by=['Topic'])
    info_df = info_df.reset_index(drop=True)
    with PdfPages(rf'{result_path}topic_female_rate_hists.pdf') as pdf:
        for index, row in df1.iterrows():
            fig = sns.histplot(data=row, x="female_rate", kde=True).set_title(
                f'Topic {info_df.iloc[index + 1]["Name"]}\n{len(row["female_rate"])} docs')
            fig = fig.get_figure()
            mean = np.nanmean(np.array(row.female_rate))
            plt.axvline(x=mean, linewidth=3, color='b', label="mean", alpha=0.5)
            pdf.savefig(fig)
            plt.clf()


def create_new_df(data_dir, data_name, loaded_model):
    # leaves docs with gender info only
    # create "female_rate" col
    # transform the model on the remaining docs and creates "topic"
    data_path = rf'{data_dir}\{data_name}'
    documents_df = pd.read_csv(data_path, encoding='utf8')
    documents_df = documents_df[~documents_df['female'].isnull()]
    documents_df = documents_df[~documents_df['male'].isnull()]
    documents_df['female_rate'] = documents_df['female'] / (documents_df['female'] + documents_df['male'])
    docs = documents_df.clean_title_and_abstract.to_list()
    topics, probs = loaded_model.transform(docs)
    col_topics = pd.Series(topics)
    # get topics
    documents_df['topic_with_outlier_topic'] = col_topics
    # convert "-1" topic to second best
    main_topic = [np.argmax(item) for item in probs]
    documents_df['major_topic'] = main_topic
    # get probs and save as str
    result_series = []
    for prob in probs:
        # add topic -1 prob - since probs sum up to the probability of not being outlier
        prob.append(1 - sum(prob))
        result_series.append(str(prob.tolist()))
    col_probs = pd.Series(result_series)
    documents_df['probs'] = col_probs
    documents_df.to_csv(rf'{data_dir}\female_rate_{data_name}',index=False)


if (__name__ == '__main__'):
    loaded_model = BERTopic.load(r"C:\Users\morfi\PycharmProjects\LDAmodeling\results\bert\bertTopic_train(81876)_n_gram_1_1_min_topic_size_50_with_probs")
    # data_dir = rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\data'
    # data_name = 'abstract_2005_2020.csv'
    # create_new_df(data_dir, data_name, loaded_model)
    data_path = rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project\data\abstract_2005_2020_gender_and_topic.csv'
    plot_topic_female_rate_hists(data_path, loaded_model)