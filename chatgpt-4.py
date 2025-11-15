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
        if train ==0:
            print(f"TRAIN: Loading {sbj}")
        elif train ==1:
            print(f"TEST: Loading {sbj}")
        else:
            print(f"VALIDATION: Loading {sbj}")

        #Load data
        brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
        image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
        #image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
        text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
        #text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

        brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
        image = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
        text  = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double') * 2.0
        label = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')

        #Preprocessing (from original colab code)
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
    #print("Final combined dataset:")
    print(features.shape, label_all.shape)

    return features, label_all



def train_model(subjects, nc=50, ac=725):
    weights = []
    for sbj in subjects:
        X, y = load_subjects([sbj], train=0, nc=nc)
        scaler = StandardScaler().fit(X)
        #scale
        X = scaler.transform(X)
        #agglomerate
        agglomeration = cluster.FeatureAgglomeration(n_clusters=ac)
        agglomeration.fit(X)
        X = agglomeration.transform(X)

        clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=200
    )
        clf.fit(X, y)
        weights.append(clf.coef_)
    weights = np.stack(weights)  # (num_subjects, n_classes, n_features)
    mean = np.mean(weights, axis=0)
    return mean, scaler, agglomeration



def finetune_model(train_mean, val_subj, nc=50, scaler=None, agglomeration=None):
    #Load the validation subject data
    X, y = load_subjects([val_subj], train=2, nc=nc)
    #Normalise 
    X = scaler.transform(X)

    #Agglomerate
    X = agglomeration.transform(X)

    #Initialise warm-start logistic regression classifer
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=False,
        warm_start=True,
        max_iter=200
    )

    #Initialize weights from training
    classes = np.unique(y).tolist()
    classes = [c-1 for c in classes]  
    clf.classes_ = classes
    clf.coef_ = train_mean[classes].copy()
    clf.intercept_ = np.zeros(len(classes))

    #Fine-tune with validation data
    clf.fit(X, y)
    #return fine-tuned model
    return clf

def test_model(model, test_subj, nc=50, scaler=None, agglomeration=None):
    X_test,  y_test  = load_subjects([test_subj], train=1, nc=nc)
    #Normalise and Agglomerate
    X_test = scaler.transform(X_test)
    X_test = agglomeration.transform(X_test)

    #Evaluation
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


#program start

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
results = {}
TEST_INDEX = 0 #0-9 - which subject to use as test
NUM_CLASSES = 100 #number up to 1654
AGGLOMERATION_COEFFICIENT=725
test_subj = subjects[TEST_INDEX]
subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]
results = []
for val_subj in subjects:
    train_subjects = [s for s in subjects if s != val_subj]
    print(f"\nUsing {val_subj} as validation subject:")
    #Training
    mean, scaler, agglomeration = train_model(train_subjects, nc=NUM_CLASSES, ac = AGGLOMERATION_COEFFICIENT)
    #Fine tuning
    model = finetune_model(mean, val_subj, nc=NUM_CLASSES, scaler=scaler, agglomeration=agglomeration)
    #Testing
    acc = test_model(model, test_subj, nc=NUM_CLASSES, scaler=scaler, agglomeration=agglomeration)
    results.append(acc)
    print(f"Accuracy for {val_subj}: {acc:.3f}")
print("AVERAGE ACCURACY:", np.mean(results))
