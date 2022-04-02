import re
import matplotlib.pyplot as plt
import csv
import math
iteration=[]
perplexity=[]
convergence=[]
coherence=[]
for line in open(r'C:\Users\katac\PycharmProjects\NLP_project\TopicModeling\LDA\src\model_callbacks.log'):
    if 'Start' in line:
        short=" ".join(line.split()[:-1])
        iteration.append(short.split()[-1])
    if 'Coherence' in line :
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
plt.xlabel("iterations")
plt.ylabel("c_v coherence value")
plt.show()