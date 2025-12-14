from sklearn.calibration import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing, svm, cluster
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, precision_recall_fscore_support, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
import mmbra
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from cwvclassifier import ClassWeightedVotingClassifier 
from sklearn.manifold import TSNE


#Data paths
data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
image_model = 'pytorch/cornet_s'
text_model = 'CLIPText'
roi = '17channels'

#Initialise variables
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
train_brain, train_image, train_text, train_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
fine_tune_brain, fine_tune_image, fine_tune_text, fine_tune_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
test_brain, test_image, test_text, test_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
NUM_CLASSES = 120
SAMPLES_PER_CLASS = 10
TRAIN_PER_CLASS = 1
FINE_TUNE_PER_CLASS = 2

num = 0 #Funky counter thingy for the for loop
for sbj in subjects:
    #Load data
    print(f"Loading {sbj}")
    brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
    image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
    #image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
    text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
    #text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

    brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
    label = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
    
    #Idk what this does but it was in the colab code and i aint touching it
    brain = brain[:, :, 20:80]  # 70ms–400ms
    brain = np.reshape(brain, (brain.shape[0], -1))

    #Only get the correct number of classes
    idx = np.squeeze(np.where(label < (NUM_CLASSES+1), True, False))
    brain, label = brain[idx, :], label[idx]
    
    #TRAINING DATA
    for i in range(NUM_CLASSES):
        start = i * SAMPLES_PER_CLASS
        end = start + SAMPLES_PER_CLASS
        train_brain[num].append(brain[start:start+TRAIN_PER_CLASS])
    num+=1 #important


for i in range(10):
    #Train data
    train_brain[i] = np.vstack(train_brain[i])
    print(np.array(train_brain[i]).shape)
print(np.array(train_brain).shape)
#X_all = np.vstack([train_brain[s] for s in range(9)])#only visualise train subjects
X_all = np.array([train_brain[s] for s in range(10)])
# print(np.array(X_all).shape)
# y_subject = [s for s in subjects if s != subjects[9]]
# scaler = StandardScaler().fit(X_all)
# X_scaled = scaler.transform(X_all)

# pca = PCA(n_components=50, random_state=42).fit(X_scaled)
# X_pca = pca.transform(X_scaled)
# X_tsne = TSNE(
#     n_components=2,
#     perplexity=30,
#     learning_rate='auto',
#     init='pca',
#     random_state=42
# ).fit_transform(X_pca)
X_flat = X_all.reshape(-1, X_all.shape[-1])
subjects = np.repeat(np.arange(X_all.shape[0]), X_all.shape[1])

# Scale (fit once)
scaler = StandardScaler().fit(X_flat)
X_scaled = scaler.transform(X_flat)

# PCA
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_scaled)

# t-SNE
X_tsne = TSNE(
    n_components=2,
    perplexity=30,
    init='pca',
    random_state=42
).fit_transform(X_pca)


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))

for s in np.unique(subjects):
    idx = subjects == s
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        s=8,
        alpha=0.6,
        label=f"sub-{s+1:02d}"
    )

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("T-SNE showing EEG feature space per Subject (120 samples)")
plt.legend(markerscale=2, fontsize=8, ncol=2, title="Subject")
plt.tight_layout()
plt.show()



# import matplotlib.pyplot as plt
# print(X_tsne.shape)
# subjects = [f"sub-{i:02d}" for i in range(1, 10)]
# indexes = list(range(9))
# plt.figure(figsize=(7, 6))
# for i in indexes:
#     idx = subjects == i
#     print(X_tsne[i, 0], X_tsne[i, 1])
#     plt.scatter(
#         X_tsne[idx, 0],
#         X_tsne[idx, 1],
#         s=8,
#         alpha=0.6,
#         label=f'{subjects[i]}'
#     )

# plt.legend(markerscale=2, fontsize=8)
# plt.title("t-SNE of Feature Space Coloured by Subject")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.tight_layout()
# plt.show()