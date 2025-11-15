import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- 1. Generate true class prototypes (10 distinct points in 2-D space)
n_classes = 10
true_prototypes = np.random.randn(n_classes, 2)

# ---- 2. Create synthetic "less-than-one-shot" samples
# Each sample = weighted mix of two true classes
n_train_samples = 5
pairs = [(i, i+1) for i in range(0, n_classes, 2)]  # (0,1), (2,3), ...
train_X = []
train_y = []

for a,b in pairs:
    mix = 0.5*true_prototypes[a] + 0.5*true_prototypes[b]
    train_X.append(mix)
    # label it with both classes: we'll expand later
    train_y.append((a,b))

train_X = np.vstack(train_X)

# ---- 3. "Expand" mixed samples into pseudo-samples for each class
expanded_X = []
expanded_y = []
for (a,b), x in zip(train_y, train_X):
    expanded_X.append(x)
    expanded_y.append(a)
    expanded_X.append(x)
    expanded_y.append(b)
expanded_X = np.vstack(expanded_X)
expanded_y = np.array(expanded_y)

# ---- 4. Train a linear classifier (with 10 classes but only 5 unique feature points!)
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
clf.fit(expanded_X, expanded_y)

# ---- 5. Evaluate on real unseen points around each true prototype
test_X = true_prototypes + 0.1*np.random.randn(*true_prototypes.shape)
test_y = np.arange(n_classes)
preds = clf.predict(test_X)

acc = accuracy_score(test_y, preds)
print(f"Less-Than-One-Shot accuracy on 10 unseen classes: {acc:.2f}")

# ---- 6. Visualize
plt.figure(figsize=(5,5))
plt.scatter(true_prototypes[:,0], true_prototypes[:,1], c=range(10), cmap='tab10', s=80, label="True classes")
plt.scatter(train_X[:,0], train_X[:,1], c='k', marker='x', s=80, label="Mixed training pts")
for i,(a,b) in enumerate(pairs):
    plt.text(train_X[i,0]+0.05, train_X[i,1], f"{a}&{b}", fontsize=9)
plt.legend()
plt.title("Less-Than-One-Shot Learning (10 classes from 5 samples)")
plt.show()
