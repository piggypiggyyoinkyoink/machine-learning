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


def load_subjects(subjects, train=0, nc=100):
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    all_brain, all_image, all_text, all_label = [], [], [], []
    NUM_CLASSES = nc
    SAMPLES_PER_CLASS = 10

    TRAIN_PER_CLASS = 5
    VALIDATION_PER_CLASS = 3

    for sbj in subjects:
        print(f"Loading {sbj} ...")

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

        idx = np.squeeze(np.where(label < (NUM_CLASSES+1), True, False))
        brain, image, text, label = brain[idx, :], image[idx, :], text[idx, :], label[idx]

        if train==0:
            #TRAINING
            for i in range(NUM_CLASSES):
                start = i * SAMPLES_PER_CLASS
                end = start + SAMPLES_PER_CLASS
                all_brain.append(brain[start:start+TRAIN_PER_CLASS])
                #print("DORKUS:",brain[start:start+TRAIN_PER_CLASS].shape)
                all_image.append(image[start:start+TRAIN_PER_CLASS])
                all_text.append(text[start:start+TRAIN_PER_CLASS])
                all_label.append(label[start:start+TRAIN_PER_CLASS])
        elif train ==1:
            #TESTING
            for i in range(NUM_CLASSES):
                start = i * SAMPLES_PER_CLASS
                end = start + SAMPLES_PER_CLASS
                all_brain.append(brain[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
                all_image.append(image[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
                all_text.append(text[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
                all_label.append(label[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
        else:
            #VALIDATION
            for i in range(NUM_CLASSES):
                start = i * SAMPLES_PER_CLASS
                end = start + SAMPLES_PER_CLASS
                all_brain.append(brain[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
                all_image.append(image[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
                all_text.append(text[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
                all_label.append(label[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
    brain_all = np.vstack(all_brain)
    image_all = np.vstack(all_image)
    text_all = np.vstack(all_text)
    label_all = np.vstack(all_label).ravel()

    features = np.hstack((brain_all, image_all, text_all))
    print("Final combined dataset:")
    print(features.shape, label_all.shape)
    #print("Agg.. features:")
    #agged_features = agg_features(features) #issue: test data is used to agg the features
    #print(agged_features.shape)
    #print(features.shape, label_all.shape)

    return features, label_all


#

def agg_features(train_features, test_features, a):
    agglo = cluster.FeatureAgglomeration(n_clusters=a)
    agglo.fit(train_features)
    agged_train = agglo.transform(train_features)
    if test_features is not None:
        agged_test = agglo.transform(test_features)
        
        return agged_train, agged_test
    else:
        return agged_train
    

def train_model(subjects, alpha=1.0,nc=50, ac=725):
    weights = []
    for sbj in subjects:
        X, y = load_subjects([sbj], train=0, nc=nc)
        scaler = StandardScaler().fit(X)
        #scale
        X = scaler.transform(X)
        #agglomerate
        X = agg_features(X, test_features=None, a=ac)

        #clf = RidgeClassifier(alpha=alpha)
        clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=200
    )
        clf.fit(X, y)
        weights.append(clf.coef_)
    weights = np.stack(weights)  # (num_subjects, n_classes, n_features)
    meta_mean = np.mean(weights, axis=0)
    return meta_mean



def finetune_model(meta_mean, test_sbj, few_shot=150, alpha=1.0, nc=50, ac=725):
    # Load validation data for the held-out subject
    X, y = load_subjects([test_sbj], train=2, nc=nc)
    #print(len(X))
    idx = np.random.choice(len(X), few_shot, replace=False)
    X_train, y_train = X[idx], y[idx]
    X_test,  y_test  = np.delete(X, idx, axis=0), np.delete(y, idx)

    # Normalize using only training data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    # Feature agglomeration
    X_train, X_test = agg_features(X_train, X_test, a=ac)

    # Use warm-start logistic regression for clean fine-tuning
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=False,
        warm_start=True,
        max_iter=200
    )
    classes = np.unique(y_train).tolist()
    classes = [c-1 for c in classes]  
    print(classes)

    # Initialize weights from the meta-learned prior
    clf.classes_ = classes
    clf.coef_ = meta_mean[classes].copy()
    clf.intercept_ = np.zeros(len(classes))

    # Fine-tune starting DIRECTLY from meta-weights
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)





subjects = [f"sub-{i:02d}" for i in range(1, 11)]
results = {}
TEST_INDEX = 0
NUM_CLASSES = 50
FEW_SHOT = 100
ALPHA = 1.0
AGGLOMERATION_COEFFICIENT=725
test_subj = subjects[TEST_INDEX]
subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]
for val_subj in subjects:
    train_subjects = [s for s in subjects if s != val_subj]
    print(f"\n=== Leave-One-Out: testing on {val_subj} ===")
    #X_train, y_train = load_subjects(train_subjects, train=0, nc=50)  
    #X_val,  y_val  = load_subjects([val_subj], train=2, nc=NUM_CLASSES)
    meta_mean = train_model(train_subjects, alpha=ALPHA, nc=NUM_CLASSES, ac = AGGLOMERATION_COEFFICIENT)
    acc = finetune_model(meta_mean, val_subj, few_shot=FEW_SHOT, alpha=ALPHA, nc=NUM_CLASSES, ac = AGGLOMERATION_COEFFICIENT)
    results[val_subj] = acc
    print(f"Accuracy for {val_subj}: {acc:.3f}")
print("\nAverage LOSO accuracy:", np.mean(list(results.values())))
