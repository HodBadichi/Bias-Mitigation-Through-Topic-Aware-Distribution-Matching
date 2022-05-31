from DistributionMatching.NoahArc.src.NoahArcCE import NoahArcCE
from DistributionMatching.NoahArc.src.NoahArcCS import NoahArcCS

""" NoahArcFactory implementation factory of 'NoahArc'
"""


class NoahArcFactory:
    @staticmethod
    def create(
            dataframe,
            similarity_metric,
            similarity_matrix,
            reset_different_topic_entries_flag,
            df_name,
            ProbabilityMatrixPath=''
    ):
        if similarity_metric == "cross_entropy":
            return NoahArcCE(
                dataframe,
                similarity_matrix,
                reset_different_topic_entries_flag,
                df_name,
                ProbabilityMatrixPath
            )
        elif similarity_metric == "cosine_similarity":
            return NoahArcCS(
                dataframe,
                similarity_matrix,
                reset_different_topic_entries_flag,
                df_name,
                ProbabilityMatrixPath
            )
        raise NotImplementedError("`NoahArcFactory` unsupported metric")
