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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier



def load_data(nc=100):
    #Data paths
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    #Initialise variables
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    train_brain, train_image, train_text, train_label = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    valid_brain, valid_image, valid_text, valid_label = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    test_brain, test_image, test_text, test_label = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    NUM_CLASSES = nc
    SAMPLES_PER_CLASS = 10
    TRAIN_PER_CLASS = 5
    VALIDATION_PER_CLASS = 3

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
        image = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
        text  = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double') * 2.0
        label = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
        #Idk what this does but it was in the colab code and i aint touching it
        brain = brain[:, :, 27:60]  # 70ms–400ms
        brain = np.reshape(brain, (brain.shape[0], -1))
        image = image[:, :100]
        #Only get the correct number of classes
        idx = np.squeeze(np.where(label < (NUM_CLASSES+1), True, False))
        brain, image, text, label = brain[idx, :], image[idx, :], text[idx, :], label[idx]
        
        #TRAINING DATA
        for i in range(NUM_CLASSES):
            start = i * SAMPLES_PER_CLASS
            end = start + SAMPLES_PER_CLASS
            train_brain[num].append(brain[start:start+TRAIN_PER_CLASS])
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
        
        num+=1 #important

    #Initialise lists
    train_features = []
    valid_features = []
    test_features = []
    train_labels = []
    valid_labels = []
    test_labels = []

    #Conglomerate brain/image/text features while maintaining subject separation
    for i in range(10):
        #Train data
        train_brain[i] = np.vstack(train_brain[i])
        train_image[i] = np.vstack(train_image[i])
        train_text[i] = np.vstack(train_text[i])
        train_label[i] = np.vstack(train_label[i])
        #add to final lists
        train_features.append(np.hstack((train_brain[i], train_image[i], train_text[i])))
        train_labels.append(train_label[i].ravel())

        #Validation data
        valid_brain[i] = np.vstack(valid_brain[i])
        valid_image[i] = np.vstack(valid_image[i])
        valid_text[i] = np.vstack(valid_text[i])
        valid_label[i] = np.vstack(valid_label[i])
        #add to final lists
        valid_features.append(np.hstack((valid_brain[i], valid_image[i], valid_text[i])))
        valid_labels.append(valid_label[i].ravel())

        #Test data
        test_brain[i] = np.vstack(test_brain[i])
        test_image[i] = np.vstack(test_image[i])
        test_text[i] = np.vstack(test_text[i])
        test_label[i] = np.vstack(test_label[i])
        #add to final lists
        test_features.append(np.hstack((test_brain[i], test_image[i], test_text[i])))
        test_labels.append(test_label[i].ravel())
    #Return shit innit
    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


def fit_preprocessors(train_features, ac=445):
    # pool all subjects’ TRAIN data
    X = np.vstack(train_features)
    #y = np.hstack(train_labels)
    #poly = PolynomialFeatures(degree=2, include_bias=False)
    #poly.fit(X)
    #X = poly.transform(X)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    agglomeration = None
    #agglomeration = LinearDiscriminantAnalysis(n_components=75).fit(X, y)
    agglomeration = cluster.FeatureAgglomeration(n_clusters=ac).fit(X)
    return scaler, agglomeration



def preprocess_data(X, scaler, agglomeration=None):
    X = scaler.transform(X)
    if agglomeration is not None:
        X = agglomeration.transform(X)
    return X



def train_model(X_train_arr, y_train_arr, num_subjects=8, scaler=None, agglomeration=None):
    weights = []
    #Fit model to each subject individually
    for i in range(num_subjects):
        #Get training data for the subject
        X, y = X_train_arr[i], y_train_arr[i]
        #Scale data
        #scaler = StandardScaler().fit(X)
        X = preprocess_data(X, scaler=scaler, agglomeration=agglomeration)


        #Fit model
        clf = LogisticRegression(
        penalty="l2",
        C=1.5,
        max_iter=200
        )
        clf.fit(X, y)
        #Yoink weights
        weights.append(clf.coef_)

    #Average weights across subjects
    weights = np.stack(weights)  
    mean = np.mean(weights, axis=0)
    #Return mean weight to be used for fine tuning
    #Return the scaler and agglomeration to be used in validation and testing
    return mean




def finetune_model(train_mean, X, y, scaler=None, agglomeration=None):
    X = preprocess_data(X, scaler=scaler, agglomeration=agglomeration)

    #Initialise warm-start logistic regression classifer
    clf = LogisticRegression(
        penalty="l2",
        C=1.5,
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
    #Return fine-tuned model
    return clf




def test_model(model, X_test, y_test, scaler=None, agglomeration=None):
    #Normalise and Agglomerate
    X_test = preprocess_data(X_test, scaler=scaler, agglomeration=agglomeration)
    #Evaluation
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)




#PROGRAM START

#Initialise variables
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
TEST_INDEX = 0 #0-9 - which subject to use as test
VALIDATION_INDEX = 0 #0-8 - which subject to use for validation
NUM_CLASSES = 100 #number up to 1654
NUM_TRAIN_SUBJECTS = 8 #there is no reason to change this, ever
COEFFICIENT_OF_AGGLOMERATION=445 #number up to 1173

test_subj = subjects[TEST_INDEX]
non_test_subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]
results = []
models = []

#Load data
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_data(nc=NUM_CLASSES)
X_test = test_features[TEST_INDEX]
y_test = test_labels[TEST_INDEX]

#Fit preprocessors for future use
scaler, agglomeration = fit_preprocessors(train_features)

#Organise data
val_subj = non_test_subjects[VALIDATION_INDEX]
X_train_arr = [(train_features[j]) for j in range(10) if j != VALIDATION_INDEX and j != TEST_INDEX]
y_train_arr = [(train_labels[j]) for j in range(10) if j != VALIDATION_INDEX and j != TEST_INDEX]
X_val = valid_features[VALIDATION_INDEX]
y_val = valid_labels[VALIDATION_INDEX]

#Training
mean = train_model(X_train_arr, y_train_arr, num_subjects=NUM_TRAIN_SUBJECTS, scaler=scaler, agglomeration=agglomeration)

#Fine tuning
model = finetune_model(mean, X_val, y_val, scaler=scaler, agglomeration=agglomeration)

#Testing
acc = test_model(model, X_test, y_test, scaler=scaler, agglomeration=agglomeration)
results.append(acc)
models.append(model)
print(f"\nTest Subject: {test_subj}")
print(f"Validation Subject: {val_subj}")
print(f"Accuracy: {acc:.3f}")
