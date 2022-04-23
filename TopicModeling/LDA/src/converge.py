import pandas as pd
import numpy as np
from gensim import models
import matplotlib.pyplot as plt
import logging
import re
import os
from gensim import corpora
### choose the callbacks classes to import
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric
from gensim.models import CoherenceModel


import matplotlib.pyplot as plt
iteration=[]
perplexity=[]
convergence=[]
coherence=[]
for line in open(r'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\logs\model_callbacks_passes_16_topics.log'):
    if 'Start' in line:
        short=" ".join(line.split()[:-1])
        iteration.append(short.split()[-1])
    if 'coherence' in line :
        coherence.append(line.split()[-1])

fields = ['iterations','cv_coherence']
# rows = zip(iteration,coherence)
# with open("conv.csv","w") as f :
#     writer = csv.writer(f)
#     writer.writerow(fields)
#     for row in rows:
#         writer.writerow(row)

iteration = [ int(i) for i in iteration]
coherence = [round(float(i),3) for i in coherence]
print(iteration)
print(coherence)
plt.plot(iteration,coherence)
plt.xlabel("passes")
plt.ylabel("c_v coherence value")
plt.title('Model Convergence ')
plt.show()