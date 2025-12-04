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



def load_data(nc=100):
    #Data paths
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    #Initialise variables
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    train_brain, train_image, train_text, train_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    valid_brain, valid_image, valid_text, valid_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    test_brain, test_image, test_text, test_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
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
        brain = brain[:, :, 20:80]  # 70ms–400ms
        image = image[:, :100]
        brain = np.reshape(brain, (brain.shape[0], -1))
        #print(f"Brain shape: {brain.shape}, Image shape: {image.shape}, Text shape: {text.shape}, Label shape: {label.shape}")

        #Only get the correct number of classes
        idx = np.squeeze(np.where(label < (NUM_CLASSES+1), True, False))
        brain, image, text, label = brain[idx, :], image[idx, :], text[idx, :], label[idx]
        
        #TRAINING DATA
        # for i in range(NUM_CLASSES):
        #     start = i * SAMPLES_PER_CLASS
        #     end = start + SAMPLES_PER_CLASS
        #     train_brain[num].append(brain[start:end])
        #     train_image[num].append(image[start:end])
        #     train_text[num].append(text[start:end])
        #     train_label[num].append(label[start:end])

        train_brain[num] = np.array(brain)
        train_image[num] = np.array(image)
        train_text[num] = np.array(text)
        train_label[num] = np.array(label)
        print(np.array(brain).shape)
        num+=1 #important
    stds = []
    lbls = []
    maxs = []
    for i in range(NUM_CLASSES*SAMPLES_PER_CLASS):
        arr = []
        lbls.append(train_label[0][i][0])
        for j in range(10):
            #print(train_brain[j].shape)
            #print(train_brain[j][i].shape)
            arr.append(train_brain[j][i])
        print(np.array(arr).shape)
        std = np.std(np.array(arr), axis=0)
        print(std.shape)
        stds.append(np.mean(std))
        maxs.append(np.max(std))
    print(np.mean(stds))
    print(np.max(maxs))
    print(np.mean(maxs))
    print(np.array(stds).shape)

    print(np.array(lbls).shape)
#PROGRAM START

#Initialise variables
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
TEST_INDEX = 0 #0-9 - which subject to use as test
NUM_CLASSES = 100 #number up to 1654

test_subj = subjects[TEST_INDEX]
non_test_subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]
results = []
models = []

#Load data
load_data(nc=NUM_CLASSES)
