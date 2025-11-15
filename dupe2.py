import torch
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



subjects = [f"sub-{i:02d}" for i in range(1, 11)]

def load_subjects(subjects):
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    # empty lists for accumulation
    all_brain, all_image, all_text, all_label = [], [], [], []

    NUM_CLASSES = 100
    SAMPLES_PER_CLASS = 10
    TRAIN_PER_CLASS = 7
    for sbj in subjects:
        print(f"Loading {sbj} ...")

        # ---- paths ----
        brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
        image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
        image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
        text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
        text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

        # ---- load data ----
        brain_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
        brain_seen = brain_seen[:, :, 27:60]   # 70–400 ms
        brain_seen = np.reshape(brain_seen, (brain_seen.shape[0], -1))
        image_seen = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
        text_seen  = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double') * 2.0
        label_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
        image_seen = image_seen[:, 0:100]

        brain_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['data'].astype('double') * 2.0
        brain_unseen = brain_unseen[:, :, 27:60]
        brain_unseen = np.reshape(brain_unseen, (brain_unseen.shape[0], -1))
        image_unseen = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))['data'].astype('double') * 50.0
        text_unseen  = sio.loadmat(os.path.join(text_dir_unseen, 'text_feat_test.mat'))['data'].astype('double') * 2.0
        label_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['class_idx'].T.astype('int')
        image_unseen = image_unseen[:, 0:100]

        # ---- filter to 50 categories ----
        idx_seen = np.squeeze(np.where(label_seen < (NUM_CLASSES+1), True, False))
        idx_unseen = np.squeeze(np.where(label_unseen < (NUM_CLASSES+1), True, False))

        brain_seen = brain_seen[idx_seen, :]
        image_seen = image_seen[idx_seen, :]
        text_seen  = text_seen[idx_seen, :]
        label_seen = label_seen[idx_seen]

        brain_unseen = brain_unseen[idx_unseen, :]
        image_unseen = image_unseen[idx_unseen, :]
        text_unseen  = text_unseen[idx_unseen, :]
        label_unseen = label_unseen[idx_unseen]

        # ---- partition within each subject (7 train, 3 test per class) ----
        all_brain.append(brain_seen)
        all_image.append(image_seen)
        all_text.append(text_seen)
        all_label.append(label_seen)


    # ---- concatenate all subjects ----
    train_brain = np.vstack(all_brain)
    train_image = np.vstack(all_image)
    train_text  = np.vstack(all_text)
    train_label = np.vstack(all_label).ravel()

    # ---- final combined features ----
    train_features = np.hstack((train_brain, train_image, train_text))

    print("Final combined dataset:")
    print(train_features.shape, train_label.shape)
    return train_features, train_label

def scale_features(train_features, test_features):
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    return train_features, test_features

"""
svm_clf = svm.SVC(kernel='linear')


brain_transformer = preprocessing.FunctionTransformer(lambda X: X[:, :561])  
brain = Pipeline(
    [('transformer_1', brain_transformer), ('brain', svm_clf)]
)


image_transformer = preprocessing.FunctionTransformer(lambda X: X[:, 561:661])  
image = Pipeline(
    [('transformer_2', image_transformer), ('image', svm_clf)]
)

text_transformer = preprocessing.FunctionTransformer(lambda X: X[:, 661:])  
text = Pipeline(
    [('transformer_3', text_transformer), ('text', svm_clf)]
)

model = StackingClassifier(
    estimators=[
        ('brain', brain),
        ('image', image),
        ('text', text),

    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    passthrough=False
)"""

model = KNeighborsClassifier(n_neighbors=1)

#Cycling through all subjects using one to test and the others as train (LOSO-CV)
#for test_subj in subjects:

#Only using sub-10 to test so that i dont have to wait an age and a half every time i run it
if True:
    test_subj = 'sub-10'
    print(f"\n=== Testing on {test_subj} ===")

    # training subjects = all except the test subject
    train_subj = [s for s in subjects if s != test_subj]

    # load data for training subjects
    X_train, y_train = load_subjects(train_subj)  # function to load/stack as before
    X_test,  y_test  = load_subjects([test_subj])
    X_train, X_test = scale_features(X_train, X_test)
    # train model
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{test_subj} accuracy: {acc:.3f}")




