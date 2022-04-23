import pandas as pd
import os
import dill
import mgzip

from DistributionMatching.NoahArc.NoahArcCE import NoahArcCE
from DistributionMatching.NoahArc.NoahArcCS import NoahArcCS
from DistributionMatching.NoahArc._NoahArcLoaded import _NoahArcLoaded
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory


class NoahArcFactory:
    @staticmethod
    def create(similarity_metric, reset_different_topic_entries_flag, similarity_matrix):
        if similarity_metric == "cross_entropy":
            return NoahArcCE(reset_different_topic_entries_flag, similarity_matrix)
        elif similarity_metric == "cosine_similarity":
            return NoahArcCS(reset_different_topic_entries_flag, similarity_matrix)
        raise NotImplementedError("`NoahArcFactory` unsupported metric")

    @staticmethod
    def save(noah_arc_object, save_path):
        database = {'matrix': noah_arc_object.probability_matrix,
                    'dataframe': noah_arc_object.documents_dataframe}
        with mgzip.open(save_path, "wb") as file:
            dill.dump(database, file)

    @staticmethod
    def load(load_path):
        with mgzip.open(load_path, "rb") as file:
            database = dill.load(file)
        probability_matrix = database['matrix']
        documents_dataframe = database['dataframe']
        return _NoahArcLoaded(probability_matrix, documents_dataframe)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.pardir, os.pardir, "data", "abstract_2005_2020_gender_and_topic.csv"))
    print(len(df))
    matrix = SimilarityMatrixFactory.create(df, similarity_metric='cross_entropy')
    Prob = NoahArcFactory.create('cross_entropy', True, matrix)
    print("loaded to memory")
    print(Prob.get_match(42))
    print(f"42 PMID is {Prob.documents_dataframe['PMID'][42]}")
    print(len(Prob.documents_dataframe))
    NoahArcFactory.save(Prob, "my_out")
    # Prob = NoahArcFactory.load("my_out")
    # print(Prob.get_match(42))
    # print(f"42 PMID is {Prob.documents_dataframe['PMID'][42]}")
    # print(len(Prob.documents_dataframe))
