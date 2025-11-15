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
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier

def load_data(nc=100):
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    all_brain, all_image, all_text, all_label = [], [], [], []
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]

    train_brain, train_image, train_text, train_label = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    valid_brain, valid_image, valid_text, valid_label = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    test_brain, test_image, test_text, test_label = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    NUM_CLASSES = nc
    SAMPLES_PER_CLASS = 10

    TRAIN_PER_CLASS = 5
    VALIDATION_PER_CLASS = 3

    num = 0
    for sbj in subjects:
        print(f"Loading {sbj}")
        

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

        
        #TRAINING DATA
        for i in range(NUM_CLASSES):
            start = i * SAMPLES_PER_CLASS
            end = start + SAMPLES_PER_CLASS
            train_brain[num].append(brain[start:start+TRAIN_PER_CLASS])
            #print("DORKUS:",brain[start:start+TRAIN_PER_CLASS].shape)
            train_image[num].append(image[start:start+TRAIN_PER_CLASS])
            train_text[num].append(text[start:start+TRAIN_PER_CLASS])
            train_label[num].append(label[start:start+TRAIN_PER_CLASS])
        #TESTING DATA
        for i in range(NUM_CLASSES):
            start = i * SAMPLES_PER_CLASS
            end = start + SAMPLES_PER_CLASS
            test_brain[num].append(brain[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
            test_image[num].append(image[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
            test_text[num].append(text[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
            test_label[num].append(label[start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS:end])
        #VALIDATION DATA
        for i in range(NUM_CLASSES):
            start = i * SAMPLES_PER_CLASS
            end = start + SAMPLES_PER_CLASS
            valid_brain[num].append(brain[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
            valid_image[num].append(image[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
            valid_text[num].append(text[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
            valid_label[num].append(label[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
        
        num+=1
    train_features = []
    valid_features = []
    test_features = []
    train_labels = []
    valid_labels = []
    test_labels = []
    for i in range(10):
        train_brain[i] = np.vstack(train_brain[i])
        train_image[i] = np.vstack(train_image[i])
        train_text[i] = np.vstack(train_text[i])
        train_features.append(np.hstack((train_brain[i], train_image[i], train_text[i])))
        #print(train_features[i].shape)
        train_label[i] = np.vstack(train_label[i])
        train_labels.append(train_label[i].ravel())
        valid_brain[i] = np.vstack(valid_brain[i])
        valid_image[i] = np.vstack(valid_image[i])
        valid_text[i] = np.vstack(valid_text[i])
        valid_features.append(np.hstack((valid_brain[i], valid_image[i], valid_text[i])))
        valid_label[i] = np.vstack(valid_label[i])
        valid_labels.append(valid_label[i].ravel())
        test_brain[i] = np.vstack(test_brain[i])
        test_image[i] = np.vstack(test_image[i])
        test_text[i] = np.vstack(test_text[i])
        test_features.append(np.hstack((test_brain[i], test_image[i], test_text[i])))
        test_label[i] = np.vstack(test_label[i])
        test_labels.append(test_label[i].ravel())
    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


def train_model(X_train_arr, y_train_arr, ac=725, num_subjects=8):
    weights = []
    for i in range(num_subjects):
        X, y = X_train_arr[i], y_train_arr[i]
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



def finetune_model(train_mean, X_val, y_val, scaler=None, agglomeration=None):
    #Load the validation subject data
    X, y = X_val, y_val
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

def test_model(model, X_test, y_test, scaler=None, agglomeration=None):
    #X_test,  y_test  = load_subjects([test_subj], train=1, nc=nc)
    #Normalise and Agglomerate
    X_test = scaler.transform(X_test)
    X_test = agglomeration.transform(X_test)

    #Evaluation
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def ensemble_train(models, X, y, ac=725, nc=100):
    #X, y = load_subjects(subjects, train=0, nc=nc)
    scaler = StandardScaler().fit(X)
    #scale
    X = scaler.transform(X)
    #agglomerate
    agglomeration = cluster.FeatureAgglomeration(n_clusters=ac)
    agglomeration.fit(X)
    X = agglomeration.transform(X)
    m = VotingClassifier(
    estimators=[('model'+str(i), models[i]) for i in range(len(models))],
    voting='hard')
    m.fit(X, y)
    return m, scaler, agglomeration

#program start

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
results = {}
TEST_INDEX = 0 #0-9 - which subject to use as test
NUM_CLASSES = 100 #number up to 1654
NUM_TRAIN_SUBJECTS = 8
AGGLOMERATION_COEFFICIENT=725
test_subj = subjects[TEST_INDEX]
non_test_subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]
results = []
models = []
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_data(nc=NUM_CLASSES)
X_test = test_features[TEST_INDEX]
y_test = test_labels[TEST_INDEX]
for i in range(len(non_test_subjects)):
    val_subj = non_test_subjects[i]
    X_train_arr = [(train_features[j]) for j in range(10) if j != i and j != TEST_INDEX]
    y_train_arr = [(train_labels[j]) for j in range(10) if j != i and j != TEST_INDEX]
    X_val = valid_features[i]
    y_val = valid_labels[i]
    #print(X_train_arr[0].shape, y_train_arr[0].shape, X_val.shape, y_val.shape)
    #breakpoint()
    train_subjects = [s for s in non_test_subjects if s != val_subj]
    #print(f"\nUsing {val_subj} as validation subject:")
    #Training
    mean, scaler, agglomeration = train_model(X_train_arr, y_train_arr, ac = AGGLOMERATION_COEFFICIENT, num_subjects=NUM_TRAIN_SUBJECTS)
    #Fine tuning
    model = finetune_model(mean, X_val, y_val, scaler=scaler, agglomeration=agglomeration)
    #Testing
    acc = test_model(model, X_test, y_test, scaler=scaler, agglomeration=agglomeration)
    results.append(acc)
    models.append(model)
    print(f"Accuracy using {val_subj} as validation subject: {acc:.3f}")
print("AVERAGE ACCURACY:", np.mean(results))

#Train ensemble model
X_train = np.vstack(X_train_arr)
y_train = np.hstack(y_train_arr)
print("AVENGERS: ENSEMBLE")
m, scaler, agglomeration = ensemble_train(models, X_train, y_train, ac=AGGLOMERATION_COEFFICIENT, nc=NUM_CLASSES)
#test ensemble model
esaccuracy = test_model(m, X_test, y_test, scaler=scaler, agglomeration=agglomeration)
print("ENSEMBLE ACCURACY:", esaccuracy) 
#accuracy with all 1654 classes: 0.6744256348246674  15/11/25  (0.6367 without ensemble)
#500 classes: 0.802 with ensemble, 0.753 without
