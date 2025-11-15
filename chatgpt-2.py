import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing, svm, cluster
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_subject_classifier(X, y, C=1.0):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = LogisticRegression(max_iter=500, C=C, multi_class='ovr')
    model.fit(X_scaled, y)
    return model, scaler


# ---------- Meta-learning loop ----------
def meta_train_classifiers(subject_ids, feature_loader, C=1.0):
    Ws = []
    for sbj in subject_ids:
        X, y = feature_loader(sbj)
        model, scaler = train_subject_classifier(X, y, C)
        Ws.append(model.coef_)
    Ws = np.stack(Ws)  # (num_subjects, n_classes, n_features)
    meta_mean = np.mean(Ws, axis=0)
    meta_cov = np.cov(Ws.reshape(Ws.shape[0], -1).T)
    return meta_mean, meta_cov


# ---------- Fine-tuning (meta-test) ----------
def meta_finetune_classifier(meta_mean, test_sbj, feature_loader, k=50, C=1.0):
    X, y = feature_loader(test_sbj)
    idx = np.random.choice(len(X), k, replace=False)
    X_train, y_train = X[idx], y[idx]
    X_test, y_test = np.delete(X, idx, axis=0), np.delete(y, idx)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize model with meta weights
    model = LogisticRegression(max_iter=500, C=C, multi_class='ovr')
    model.fit(X_train, y_train)
    model.coef_ = meta_mean.copy()  # start from meta-prior
    model.fit(X_train, y_train)     # fine-tune

    acc = model.score(X_test, y_test)
    return acc

# ---------- Example feature loader ----------
def feature_loader(sbj):
    # Example paths: adapt for your real setup
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'


    # load EEG, image, text features
    brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
    image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
    #image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
    text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
    #text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

    brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
    image = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
    text  = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double') * 2.0
    label = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')

    # Preprocess
    brain = brain[:, :, 27:60]  # 70–400 ms
    brain = np.reshape(brain, (brain.shape[0], -1))
    image = image[:, :100]
    NUM_CLASSES = 20

    idx = np.squeeze(np.where(label < (NUM_CLASSES+1), True, False))
    brain, image, text, label = brain[idx, :], image[idx, :], text[idx, :], label[idx]

    # combine modalities
    X = np.hstack((brain, image, text))

    # target could be image embedding itself (EEG→Image)
    Y = label
    return X, Y

# ---------- Run Meta-Learning ----------
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
train_subjects = subjects[:-1]
test_subject = subjects[-1]

meta_mean, meta_cov = meta_train_classifiers(train_subjects, feature_loader)
acc = meta_finetune_classifier(meta_mean, test_subject, feature_loader)
print(f"Meta-learned classification accuracy on {test_subject}: {acc:.3f}")