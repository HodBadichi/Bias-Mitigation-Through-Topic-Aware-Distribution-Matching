import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from DistributionMatching.Utils.Config import config


def AreWomenMinority(document_index, dataframe, bias_by_topic=True):
    """
        Determines whether the ith document has a decent women ratio out of the whole participants
        :param document_index:Int, the document index we want to determine if has women minority
        :param dataframe: pandas dataframe , the whole documents dataframe
        :param bias_by_topic:Bool, whether to allow or not allow matches between documents from common topic in case its
        True we calculate the threshold according to the document`s topic female rate mean
        :return:Bool, True in case there is women minority in the document
    """
    threshold = config['women_minority_threshold']
    female_rate = dataframe.iloc[document_index]["female_rate"]

    if bias_by_topic is True:
        topic = dataframe.iloc[document_index]["major_topic"]
        threshold = dataframe.loc[dataframe['major_topic'] == topic]['female_rate'].mean()

    # edge cases
    if female_rate == 0:
        return True
    elif female_rate == 1:
        return False
    elif female_rate < threshold:
        return True
    else:
        return False


def PlotTopicFemaleRateHists(data_path, loaded_model, result_path=""):
    """
    Plots each topic female ratio distribution
        :param data_path:Path, the dataframe path
        :param loaded_model: BertTopicModel, the chosen bertopic model where we get the topics distrubition from
        :param result_path:Path, where to save the PDF results
    """
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
