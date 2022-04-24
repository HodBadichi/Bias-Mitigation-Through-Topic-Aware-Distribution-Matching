import torch
import matplotlib.pyplot as plt
from DistributionMatching.NoahArc.NoahArcFactory import NoahArcFactory
import pandas as pd
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory
from DistributionMatching.utils import config


def plot_document_matches_distribution():
    dataframe = pd.read_csv(
        r'C:\Users\katac\PycharmProjects\NLP_project_v2\data\abstract_2005_2020_gender_and_topic.csv')
    sim_matrix = SimilarityMatrixFactory.create(dataframe,
                                                "cross_entropy")
    print("loaded similarity matrix")
    noah_arc = NoahArcFactory.create("cross_entropy", True, sim_matrix)
    # NoahArcFactory.save(noah_arc,"ok")
    # noah_arc = NoahArcFactory.load("ok")
    print("loaded Noah arc model")
    results = {}
    for idx, row in enumerate(noah_arc._similarity_matrix):
        number_of_possible_matches = torch.count_nonzero(row).item()
        if number_of_possible_matches in results.keys():
            if number_of_possible_matches == 21784:
                print(f"Motherfucking problematic document is {idx} ")
            results[number_of_possible_matches] += 1
        else:
            results[number_of_possible_matches] = 1
    print(f"max key is {max(list(results.keys()))}")
    results = {k:v for k,v in results.items() if k <= 10}
    plt.xlabel("Possible matched documents")
    plt.ylabel("Number of documents")
    plt.bar(list(results.keys()), results.values(), color='g')
    plt.show()

plot_document_matches_distribution()
