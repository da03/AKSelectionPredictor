import torch
import numpy as np

data = torch.load('dataset.pt')

import random
random.seed(1234)

precisions = []
recalls = []
repeats = 10
for p in np.linspace(0, 1.0, num=1000):
    tp = 0
    data_positive = 0
    pred_positive = 0
    for _ in range(repeats):
        for x, y in data['test']:
            q = random.random()
            if q <= p:
                predicted = True
            else:
                predicted = False
            if predicted and y:
                tp += 1
            if y:
                data_positive += 1
            if predicted:
                pred_positive += 1
    precision = tp / max(1e-6, pred_positive)
    recall = tp / max(1e-6, data_positive)
    precisions.append(precision)
    recalls.append(recall)

import matplotlib.pyplot as plt
plt.plot(precisions, recalls, '.')
plt.xlabel('precision')
plt.ylabel('recall')
plt.savefig('baseline_random.png')
