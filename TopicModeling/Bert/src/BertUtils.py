from bertopic import BERTopic
from matplotlib import pyplot as plt


def BertShowTopicFrequency(model_path, model_name):
    """
    Create a bar chart of frequency of topics (major topic) among all documents
    :param model_path:BertTopic model path to visualize
    :param model_name: BertTopic model name to visualize
    :return: None
    """
    loaded_model = BERTopic.load(model_path)
    frequency = loaded_model.get_topic_freq()
    topic_freq = {}
    for index, row in frequency.iterrows():
        if row[0] != -1:
            topic_freq[row[0]] = row[1]
        else:
            count_general_topic = row[1]
    plt.bar(list(topic_freq.keys()), topic_freq.values(), color='g')
    plt.title(f"{model_name}\nTopic -1 with {count_general_topic} documents")
    plt.xlabel("Topic Number")
    plt.ylabel("Frequency among documents")
    plt.show()
