WOMEN_MINORITY_THRESHOLD = 0.5
DATAFRAME_PATH= rf'..\data\abstract_2005_2020_gender_and_topic.csv'


def printTensorNonZeros(my_tensor):
    indices = [i.item() for i in list((my_tensor!= 0).nonzero())]
    print(my_tensor[indices])

def getTensorZeros(my_tensor):
    return [i[0].item() for i in(my_tensor == 0).nonzero()]


def AreWomenMinority(document_index, dataframe, bias_by_topic=True):
    threshold = WOMEN_MINORITY_THRESHOLD
    female_rate = dataframe.iloc[document_index]["female_rate"]

    if bias_by_topic is True:
        topic = dataframe.iloc[document_index]["major_topic"]
        threshold = dataframe.loc[dataframe['major_topic'] == topic]['female_rate'].mean()

    if (female_rate <= threshold):
        return True
    else:
        return False

def GetFemaleRatePerTopicMean(dataframe):
    topic_to_female_rate = dataframe.groupby('major_topic')['female_rate'].mean()
    return topic_to_female_rate