def LoadAbstractPubMedData():
    return pd.read_csv(config['data']['full'], encoding='utf8')

def LoadTopicModel():
    return BERTopic.load(config['models']['topic_model_path'])