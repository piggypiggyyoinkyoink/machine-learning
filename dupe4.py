from sklearn.linear_model import RidgeClassifier
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



subjects = [f"sub-{i:02d}" for i in range(1, 11)]


def load_subjects(subjects, train=True, nc=100):
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    all_brain, all_image, all_text, all_label = [], [], [], []
    NUM_CLASSES = nc
    SAMPLES_PER_CLASS = 10

    TRAIN_PER_CLASS = 7

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

        # For seen data, split into train/test within subject
        if train:
            #for training data take the first 7 samples of each class for each subject to prevent data leakage
            for i in range(NUM_CLASSES):
                start = i * SAMPLES_PER_CLASS
                end = start + SAMPLES_PER_CLASS
                all_brain.append(brain[start:start+TRAIN_PER_CLASS])
                #print("DORKUS:",brain[start:start+TRAIN_PER_CLASS].shape)
                all_image.append(image[start:start+TRAIN_PER_CLASS])
                all_text.append(text[start:start+TRAIN_PER_CLASS])
                all_label.append(label[start:start+TRAIN_PER_CLASS])
        else:
            #for testing take the last 3
            for i in range(NUM_CLASSES):
                start = i * SAMPLES_PER_CLASS
                end = start + SAMPLES_PER_CLASS
                all_brain.append(brain[start+TRAIN_PER_CLASS:end])
                all_image.append(image[start+TRAIN_PER_CLASS:end])
                all_text.append(text[start+TRAIN_PER_CLASS:end])
                all_label.append(label[start+TRAIN_PER_CLASS:end])
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

def scale_features(train_features, test_features):
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    return train_features, test_features

def agg_features(train_features, test_features, a):
    agglo = cluster.FeatureAgglomeration(n_clusters=a)
    agglo.fit(train_features)
    agged_train = agglo.transform(train_features)
    agged_test = agglo.transform(test_features)
    return agged_train, agged_test




AGGLOMERATION_COEFFICIENT=726
knn_clf = KNeighborsClassifier(n_neighbors=1)

transformer_eins = preprocessing.FunctionTransformer(lambda X: X[:, :(AGGLOMERATION_COEFFICIENT//3)])  
eins = Pipeline(
    [('transformer_1', transformer_eins), ('eins', knn_clf)]
)


transformer_zwei = preprocessing.FunctionTransformer(lambda X: X[:, (AGGLOMERATION_COEFFICIENT//3):(2*(AGGLOMERATION_COEFFICIENT//3))])  
zwei = Pipeline(
    [('transformer_2', transformer_zwei), ('zwei', knn_clf)]
)

transformer_drei = preprocessing.FunctionTransformer(lambda X: X[:, (2*(AGGLOMERATION_COEFFICIENT//3)):])  
drei = Pipeline(
    [('transformer_3', transformer_drei), ('drei', knn_clf)]
)

dingus = KNeighborsClassifier(n_neighbors=2)

model = StackingClassifier( #all this bs increases accuracy by 0.002 
    estimators=[
        ('knn', dingus),
        ('eins', eins),
        ('zwei', zwei),
        ('drei', drei),
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    passthrough=False
)
model = KNeighborsClassifier(n_neighbors=1)

model = RidgeClassifier(alpha=1.0)

accs = []
#Cycling through all subjects using one to test and the others as train (LOSO-CV)
#for test_subj in subjects:

#Only using sub-10 to test so that i dont have to wait an age and a half every time i run it
if True:
    test_subj = 'sub-10'


    print(f"\nTesting on {test_subj} ")
    # training subjects = all except the test subject
    train_subj = [s for s in subjects if s != test_subj]
    # load data for training subjects
    X_train, y_train = load_subjects(train_subj, nc=50)  # function to load/stack as before
    X_test,  y_test  = load_subjects([test_subj], train=False, nc=50)
    X_train, X_test = scale_features(X_train, X_test)
    X_train, X_test = agg_features(X_train, X_test, AGGLOMERATION_COEFFICIENT)
    print(X_train.shape, X_test.shape)

    # train model
    model.fit(X_train, y_train)

    # evaluate
    """
    print("Unique labels in train:", np.unique(y_train))
    print("Unique labels in test: ", np.unique(y_test))
    # create hashes of each row to check overlap quickly
    train_hashes = set(map(lambda r: hash(r.tobytes()), X_train))
    test_hashes = set(map(lambda r: hash(r.tobytes()), X_test))
    overlap = len(train_hashes & test_hashes)
    print("Number of identical samples in train ∩ test:", overlap)  # should be 0
    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_test ", X_test.shape,  "y_test ", y_test.shape)

    """

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accs.append(acc)
    print(f"{test_subj} accuracy: {acc:.3f}")

print("Average accuracy: ", (sum(accs)/len(accs)))

#currently 0.843 accuracy with just testing on sub-10 with 200 classes.
# currently 0.897 average accuracy with 100 classes
