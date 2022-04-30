import sys
sys.path.append('/home/mor.filo/nlp_project/')
import torch
import matplotlib.pyplot as plt
from DistributionMatching.NoahArc.NoahArcFactory import NoahArcFactory
import pandas as pd
from DistributionMatching.SimilarityMatrix.SimilarityMatrixFactory import SimilarityMatrixFactory
import os

def plot_document_matches_distribution():
    dataframe = pd.read_csv(
        rf'C:\Users\{os.getlogin()}\PycharmProjects\NLP_project_v2\data\abstract_2005_2020_gender_and_topic.csv')
    sim_matrix = SimilarityMatrixFactory.create(dataframe,
                                                "cross_entropy")
    noah_arc = NoahArcFactory.create("cross_entropy", sim_matrix, True)
    # NoahArcFactory.save(noah_arc,"ok")
    # noah_arc = NoahArcFactory.load("ok")
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

    results_5 = 0
    results_10 = 0

    for k,v in results.items():
        if k<=5:
            results_5+=v
            results_10 +=v
        elif k<=10:
            results_10+=v
    print(f"Disposed documents when 5 is threshold {results_5}")
    print(f"Disposed documents when 10 is threshold {results_10}")

plot_document_matches_distribution()
