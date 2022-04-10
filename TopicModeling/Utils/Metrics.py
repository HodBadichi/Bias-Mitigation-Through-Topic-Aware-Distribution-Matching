from abc import ABCMeta, abstractmethod
from TopicModeling.Utils.my_exceptions import CustomError
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import pandas as pd
import gensim.models


class Metrics(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, model, metrics_list):
        self.model = model
        self.metrics_list = metrics_list

    @abstractmethod
    def _measure(self, metric):
        pass

    def evaluate(self, metric):
        if metric not in self.metrics_list:
            raise CustomError("Tried to evaluate a non-existing Metric")
        return self._measure(metric)

    def evaluate_all_metrics(self):
        evaluations_dict = {}
        for metric in self.metrics_list:
            evaluations_dict[metric] = self._measure(metric)
            print(f"{metric}: {evaluations_dict[metric]}")
        return evaluations_dict


class LDAMetrics(Metrics):
    metrics_list = ["perplexity", "c_v", "u_mass", "c_npmi", "c_uci"]

    def __init__(self, model, curr_corpus, curr_texts):
        super().__init__(model, self.metrics_list)
        self.corpus = curr_corpus
        self.texts = curr_texts

    def _measure(self, metric):
        if (metric == "perplexity"):
            return self.model.log_perplexity(self.corpus)
        else:
            coherencemodel = gensim.models.CoherenceModel(
                model=self.model, texts=self.texts, corpus=self.corpus,coherence=metric)
            return coherencemodel.get_coherence()


class BertTopicMetrics(Metrics):
    metrics_list = ["c_v", "u_mass", "c_npmi", "c_uci"]

    def __init__(self, model, docs, topics):
        super().__init__(model, self.metrics_list)
        self.model = model
        self.docs = docs
        self.topics = topics
        documents = pd.DataFrame({"Document": self.docs,
                                  "ID": range(len(self.docs)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names()
        self.tokens = [analyzer(doc) for doc in cleaned_docs]
        self.dictionary = corpora.Dictionary(self.tokens)
        self.corpus = [self.dictionary.doc2bow(token) for token in self.tokens]
        self.topic_words = [[words for words, _ in self.model.get_topic(topic)]
                       for topic in range(len(set(self.topics)) - 1)]


    def _measure(self, metric):
        # Evaluate
        coherence_model = CoherenceModel(topics=self.topic_words,
                                         texts=self.tokens,
                                         corpus=self.corpus,
                                         dictionary=self.dictionary,
                                         coherence=metric)
        return coherence_model.get_coherence()
