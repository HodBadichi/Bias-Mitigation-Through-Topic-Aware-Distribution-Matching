import yaml
from yaml import SafeLoader

with open('my_config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


def AreWomenMinority(document_index, dataframe, bias_by_topic=True):
    threshold = config['women_minority_threshold']
    female_rate = dataframe.iloc[document_index]["female_rate"]

    if bias_by_topic is True:
        topic = dataframe.iloc[document_index]["major_topic"]
        threshold = dataframe.loc[dataframe['major_topic'] == topic]['female_rate'].mean()

    if female_rate <= threshold:
        return True
    else:
        return False


def GetFemaleRatePerTopicMean(dataframe):
    topic_to_female_rate = dataframe.groupby('major_topic')['female_rate'].mean()
    return topic_to_female_rate
