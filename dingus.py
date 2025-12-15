from collections import defaultdict
import matplotlib.pyplot as plt


outgoing_errors = defaultdict(int)  # A -> *
incoming_errors = defaultdict(int)  # * -> B
c = defaultdict(int)

#with open("./visualisations_baseline/confused_pairs.txt") as f:
with open("./visualisations_improved/confused_pairs.txt") as f:
    for line in f:
        left, count = line.split(":")
        count = int(count.strip())

        true_cls, pred_cls = map(str.strip, left.split("->"))
        c[pred_cls] += count
        outgoing_errors[true_cls] += count
        incoming_errors[pred_cls] += count

# Compute predictions per class
predictions_per_class = {}

all_classes = set(outgoing_errors) | set(incoming_errors)

for cls in all_classes:
    tp = 2 - outgoing_errors.get(cls, 0)
    fp = incoming_errors.get(cls, 0)
    predictions_per_class[cls] = int(tp + fp)

import numpy as np

vals = np.array(list(predictions_per_class.values()))
unique, counts = np.unique(vals, return_counts=True)
unique = unique.tolist()
counts= counts.tolist()
dingus = dict(zip(unique, counts))
print(c)
print(unique)
print(counts)
print(vals.min(), vals.mean(), vals.max())

plt.bar(unique, counts) 
plt.title("Distribution of class predictions")
plt.xlabel("Number of predictions for class")
plt.ylim(top=500)
plt.xticks(range(0,17))
plt.ylabel("Number of classes")
plt.show()
