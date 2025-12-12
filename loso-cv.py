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
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
from cwvclassifier import ClassWeightedVotingClassifier

def load_data(nc=100):
    #Data paths
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    #Initialise variables
    subjects = [F"sub-{i:02d}" for i in range(1, 11)]
    train_brain, train_image, train_text, train_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    ft_brain, ft_image, ft_text, ft_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    test_brain, test_image, test_text, test_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    NUM_CLASSES = nc
    SAMPLES_PER_CLASS = 10
    TRAIN_PER_CLASS = 5
    VALIDATION_PER_CLASS = 3
    TEST_PER_CLASS = 2

    num = 0 #Funky counter thingy for the for loop
    for sbj in subjects:
        #Load data
        print(F"Loading {sbj}")
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
        brain = brain[:, :, 20:80]  # 70ms–400ms
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

        #FT DATA
        for i in range(NUM_CLASSES):
            start = i * SAMPLES_PER_CLASS
            end = start + SAMPLES_PER_CLASS
            ft_brain[num].append(brain[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
            ft_image[num].append(image[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
            ft_text[num].append(text[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
            ft_label[num].append(label[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+VALIDATION_PER_CLASS])
        
        num+=1 #important

    
    #Initialise lists
    train_features = []
    ft_features = []
    test_features = []
    train_labels = []
    ft_labels = []
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

        #Fine-tune data
        ft_brain[i] = np.vstack(ft_brain[i])
        ft_image[i] = np.vstack(ft_image[i])
        ft_text[i] = np.vstack(ft_text[i])
        ft_label[i] = np.vstack(ft_label[i])
        #add to final lists
        ft_features.append(np.hstack((ft_brain[i], ft_image[i], ft_text[i])))
        ft_labels.append(ft_label[i].ravel())

        #Test data
        test_brain[i] = np.vstack(test_brain[i])
        test_image[i] = np.vstack(test_image[i])
        test_text[i] = np.vstack(test_text[i])
        test_label[i] = np.vstack(test_label[i])
        #add to final lists
        test_features.append(np.hstack((test_brain[i], test_image[i], test_text[i])))
        test_labels.append(test_label[i].ravel())
    #Return
    return train_features, train_labels, ft_features, ft_labels, test_features, test_labels


def fit_preprocessors(train_features, ac=445):
    # pool all subjects’ TRAIN data
    X = np.vstack(train_features)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    agglomeration = None
    agglomeration = cluster.FeatureAgglomeration(n_clusters=ac).fit(X)
    return scaler, agglomeration



def preprocess_data(X, scaler, agglomeration=None):
    X = scaler.transform(X)
    if agglomeration is not None:
        X = agglomeration.transform(X)
    return X



def train_model(X_train_arr, y_train_arr, num_subjects=7, scaler=None, agglomeration=None):
    clfs = []

    #Fit model to each subject individually
    for i in range(num_subjects):
        #Get training data for the subject
        X, y = X_train_arr[i], y_train_arr[i]
        #Preproccess data
        X = preprocess_data(X, scaler=scaler, agglomeration=agglomeration)

        #Fit model
        clf = LogisticRegression(
        penalty="l2",
        C=1.5,
        max_iter=200
        )
        clf.fit(X, y)
        #Yoink weights
        clfs.append(clf)
    
    return clfs



def finetune_model(models, X, y, scaler=None, agglomeration=None, alpha=1.0):
    X = preprocess_data(X, scaler=scaler, agglomeration=agglomeration)

    m = ClassWeightedVotingClassifier(
    estimators=[('model'+str(i), models[i]) for i in range(len(models))],
    alpha=alpha,              
    weighted=True
    )
    m.fit(X, y)
    return m




def test_model(model, X_test, y_test, scaler=None, agglomeration=None):
    #preprocess data
    X_test = preprocess_data(X_test, scaler=scaler, agglomeration=agglomeration)
    #Evaluation
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)



#PROGRAM START

#Initialise variables
subjects = [F"sub-{i:02d}" for i in range(1, 11)]
TEST_INDEX = 0 #0-9 - which subject to use as test
NUM_CLASSES = 100 #number up to 1654
NUM_TRAIN_SUBJECTS = 7 #there is no reason to change this, ever
F=565 #number up to 1632
ALPHA=1.0

test_subj = subjects[TEST_INDEX]
non_test_subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]
#Load data
train_features, train_labels, ft_features, ft_labels, test_features, test_labels = load_data(nc=NUM_CLASSES)

#Leave One Subject Out Cross-validation over non-test subjects
#for F in [560, 565, 570, 580, 585, 590]:
for ALPHA in [0, 2.25, 2.75, 2.3, 2.6, 2.8]:
    results = []
    models = []

    for i in range(len(non_test_subjects)):
        #Organise data
        val_subj = non_test_subjects[i]
        val_idx = subjects.index(val_subj)
        train_subjects_index = [subjects.index(s) for s in non_test_subjects if s != val_subj]
        X_train_arr = [(train_features[j]) for j in train_subjects_index]
        y_train_arr = [(train_labels[j]) for j in train_subjects_index]
        scaler, agglomeration = fit_preprocessors(X_train_arr, ac=F)
        X_ft = ft_features[val_idx]
        y_ft = ft_labels[val_idx]
        X_val = test_features[val_idx]
        y_val = test_labels[val_idx]

        #Training
        modelsList = train_model(X_train_arr, y_train_arr, num_subjects=NUM_TRAIN_SUBJECTS, scaler=scaler, agglomeration=agglomeration)
        
        #Fine tuning
        model = finetune_model(modelsList, X_ft, y_ft, scaler=scaler, agglomeration=agglomeration, alpha=ALPHA)
        #Testing
        acc = test_model(model, X_val, y_val, scaler=scaler, agglomeration=agglomeration)
        results.append(acc)
        models.append(model)
        #print(F"Accuracy using {val_subj} as validation subject: {acc:.3f}")
    #Print average accuracy across all validation subjects    
    #print("F =", F)
    print("ALPHA =", ALPHA)
    print("AVERAGE ACCURACY:", np.mean(results))


        