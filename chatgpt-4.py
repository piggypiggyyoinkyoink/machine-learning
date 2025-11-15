from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
from sklearn.linear_model import RidgeClassifier

# ---------- FEATURE LOADER (EEG + Image + Text) ----------
def feature_loader(sbj):
    data_root = './data/ThingsEEG-Text'
    roi = '17channels'
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'

    # ---- EEG ----
    brain_dir = os.path.join(data_root, 'brain_feature', roi, sbj)
    eeg_mat = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))
    X_brain = eeg_mat['data'].astype('double') * 2.0
    X_brain = X_brain[:, :, 27:60]   # 70–400 ms
    X_brain = X_brain.reshape(X_brain.shape[0], -1)
    labels = eeg_mat['class_idx'].T.astype('int').ravel()

    # ---- Image ----
    image_dir = os.path.join(data_root, 'visual_feature/ThingsTrain', image_model, sbj)
    X_image = sio.loadmat(os.path.join(image_dir, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
    X_image = X_image[:, :100]

    # ---- Text ----
    text_dir = os.path.join(data_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
    X_text = sio.loadmat(os.path.join(text_dir, 'text_feat_train.mat'))['data'].astype('double') * 2.0

    # ---- Combine all modalities ----
    X = np.hstack((X_brain, X_image, X_text))

    # ---- Limit to 50 categories (optional) ----
    NUM_CLASSES = 50
    valid_idx = np.where(labels < NUM_CLASSES+1)[0]
    X = X[valid_idx]
    y = labels[valid_idx]
    return X, y

def agg_features(train_features, test_features, a):
    agglo = cluster.FeatureAgglomeration(n_clusters=a)
    agglo.fit(train_features)
    agged_train = agglo.transform(train_features)
    if test_features is not None:
        agged_test = agglo.transform(test_features)
        
        return agged_train, agged_test
    else:
        return agged_train
    
# ---------- SUBJECT-LEVEL TRAINING ----------
def train_subject_classifier(X, y, alpha=1.0):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xs = agg_features(X, test_features=None, a=725)

    clf = RidgeClassifier(alpha=alpha)
    clf.fit(Xs, y)
    return clf, scaler


# ---------- META-TRAINING ----------
def meta_train_classifiers(subjects, feature_loader, alpha=1.0):
    weights = []
    for sbj in subjects:
        X, y = feature_loader(sbj)
        clf, _ = train_subject_classifier(X, y, alpha)
        weights.append(clf.coef_)
    weights = np.stack(weights)  # (num_subjects, n_classes, n_features)
    meta_mean = np.mean(weights, axis=0)
    return meta_mean


# ---------- META-TEST / FINE-TUNE ----------
def meta_finetune_classifier(meta_mean, test_sbj, feature_loader, few_shot=100, alpha=1.0):
    X, y = feature_loader(test_sbj)
    idx = np.random.choice(len(X), few_shot, replace=False)
    X_train, y_train = X[idx], y[idx]
    X_test, y_test = np.delete(X, idx, axis=0), np.delete(y, idx)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_test = agg_features(X_train, X_test, 725)

    clf = RidgeClassifier(alpha=alpha)
    clf.fit(X_train, y_train)
    clf.coef_ = meta_mean.copy()     # start from meta-prior weights
    clf.fit(X_train, y_train)        # fine-tune

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


# ---------- LOSO META-LEARNING ----------
def loso_meta_learning(subjects, feature_loader, few_shot, alpha):
    results = {}
    for test_sbj in subjects:
        train_subjects = [s for s in subjects if s != test_sbj]
        print(f"\n=== Leave-One-Out: testing on {test_sbj} ===")
        meta_mean = meta_train_classifiers(train_subjects, feature_loader, alpha)
        acc = meta_finetune_classifier(meta_mean, test_sbj, feature_loader, few_shot, alpha)
        results[test_sbj] = acc
        print(f"Accuracy for {test_sbj}: {acc:.3f}")
    print("\nAverage LOSO accuracy:", np.mean(list(results.values())))
    return results


# ---------- RUN ----------
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
results = loso_meta_learning(subjects, feature_loader, few_shot=300, alpha=1.0)

#TODO: use loader from dupe4.py to ensure proper separation of training and test data.