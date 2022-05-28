import sys
import torch
import os

if os.name != 'nt':
    sys.path.append('/home/mor.filo/nlp_project/')

import matplotlib.pyplot as plt
import pandas as pd

from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory
from DistributionMatching.NoahArc.NoahArcFactory import NoahArcFactory


def plot_document_matches_distribution():
    dataframe = pd.read_csv(
        rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project_v2\data\abstract_2005_2020_gender_and_topic.csv'
    )

    dataframe = dataframe.loc[dataframe['belongs_to_group'] == 'test']
    # test_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'test']
    # val_df = self.documents_df.loc[self.documents_df['belongs_to_group'] == 'val']
    dataframe = dataframe.reset_index()

    sim_matrix = SimilarityMatrixFactory.create(dataframe, "cross_entropy", 'test')
    noah_arc = NoahArcFactory.create(dataframe, "cross_entropy", sim_matrix, True, 'test')
    # torch.save(noah_arc.probability_matrix, "ok_val")
    results = {}
    for idx, row in enumerate(noah_arc._similarity_matrix):
        number_of_possible_matches = torch.count_nonzero(row).item()
        if number_of_possible_matches in results.keys():
            results[number_of_possible_matches] += 1
        else:
            results[number_of_possible_matches] = 1
    print(f"max key is {max(list(results.keys()))}")

    plt.figure(1)
    plt.title("Document matching with 'Topic' and 'Bias' resets")
    plt.xlabel("Possible matched documents")
    plt.ylabel("Number of documents")
    temp = {k: v for k, v in results.items() if k <= 10}
    plt.bar(list(temp.keys()), temp.values(), color='g')
    plt.show()

    plt.figure(2)
    plt.title("Document matching with 'Topic' and 'Bias' resets")
    plt.xlabel("Possible matched documents")
    plt.ylabel("Number of documents")
    temp = {k: v for k, v in results.items() if k <= 50}
    plt.bar(list(temp.keys()), temp.values(), color='g')
    plt.show()

    plt.figure(3)
    plt.title("Document matching with 'Topic' and 'Bias' resets")
    plt.xlabel("Possible matched documents")
    plt.ylabel("Number of documents")
    temp = {k: v for k, v in results.items() if k <= 100}
    temp = {k: v for k, v in temp.items() if k % 10 == 0}
    plt.bar(list(temp.keys()), temp.values(), color='g')
    plt.show()

    plt.figure(4)
    plt.title("Document matching with 'Topic' and 'Bias' resets")
    plt.xlabel("Possible matched documents")
    plt.ylabel("Number of documents")
    plt.bar(list(results.keys()), results.values(), color='g')
    plt.show()

    plt.figure(5)
    plt.title("Document matching with 'Topic' and 'Bias' resets")
    plt.xlabel("Possible matched documents")
    plt.ylabel("Number of documents")
    temp = {k: v for k, v in results.items() if k <= 1000}
    temp = {k: v for k, v in temp.items() if k % 10 == 0}
    plt.bar(list(temp.keys()), temp.values(), color='g')
    plt.show()

    results_list = [0] * 11

    for index, current_thresh in enumerate(results_list):
        for k, v in results.items():
            if k <= index:
                results_list[index] += v
    print(f"test df size is : {len(dataframe)}")
    for index, item in enumerate(results_list):
        print(f"Disposed values from test df when {index} is threshold: {results_list[index]}")


if __name__ == '__main__':
    plot_document_matches_distribution()
