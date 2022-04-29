import pandas as pd
from bertopic import BERTopic
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import yaml


# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        loaded_config = yaml.safe_load(file)
    return loaded_config


config = load_config("../PubMedConfig.yaml")


def are_women_minority(document_index, dataframe, bias_by_topic=True):
    threshold = config['women_minority_threshold']
    female_rate = dataframe.iloc[document_index]["female_rate"]

    if bias_by_topic is True:
        topic = dataframe.iloc[document_index]["major_topic"]
        threshold = dataframe.loc[dataframe['major_topic'] == topic]['female_rate'].mean()

    #edge cases
    if female_rate == 0:
        return True
    elif female_rate == 1:
        return False
    elif female_rate < threshold:
        return True
    else:
        return False


def load_abstract_PubMedData():
    return pd.read_csv(config['data']['full'], encoding='utf8')


def load_topic_model():
    return BERTopic.load(config['models']['topic_model_path'])


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


