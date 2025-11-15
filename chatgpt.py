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

# -----------------------------
# Shared data loader
# -----------------------------
def load_subjects(subjects, use_seen=True, num_classes=50):
    """
    Loads EEG, image, and text features for one or more subjects.
    Args:
        subjects: list of subject IDs (e.g. ['sub-01', 'sub-02'])
        use_seen: True → load seen data; False → load unseen data
        num_classes: how many classes to include (default 50)
    Returns:
        features (numpy array), labels (numpy array)
    """
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    all_brain, all_image, all_text, all_label = [], [], [], []
    samples_per_class = 10
    num_train_per_class = 7

    for sbj in subjects:
        print(f"Loading {sbj} ({'seen' if use_seen else 'unseen'}) ...")

        brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
        image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
        image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
        text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
        text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

        if use_seen:
            brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
            image = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
            text = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double') * 2.0
            label = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
        else:
            brain = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['data'].astype('double') * 2.0
            image = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))['data'].astype('double') * 50.0
            text = sio.loadmat(os.path.join(text_dir_unseen, 'text_feat_test.mat'))['data'].astype('double') * 2.0
            label = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['class_idx'].T.astype('int')

        # Preprocess
        brain = brain[:, :, 27:60]   # 70–400 ms
        brain = np.reshape(brain, (brain.shape[0], -1))
        image = image[:, :100]

        # Filter to chosen classes
        idx = np.squeeze(np.where(label < (num_classes + 1), True, False))
        brain, image, text, label = brain[idx, :], image[idx, :], text[idx, :], label[idx]

        # Partition (7 train / 3 test per class)
        if use_seen:
            for i in range(num_classes):
                start = i * samples_per_class
                end = start + samples_per_class
                all_brain.append(brain[start:start+num_train_per_class])
                all_image.append(image[start:start+num_train_per_class])
                all_text.append(text[start:start+num_train_per_class])
                all_label.append(label[start:start+num_train_per_class])
        else:
            # For unseen (zero-shot) data, take all
            all_brain.append(brain)
            all_image.append(image)
            all_text.append(text)
            all_label.append(label)

    # Combine all subjects
    brain_all = np.vstack(all_brain)
    image_all = np.vstack(all_image)
    text_all = np.vstack(all_text)
    label_all = np.vstack(all_label).ravel()

    features = np.hstack((brain_all, image_all, text_all))
    return features, label_all


# -----------------------------
# Utility: scale data
# -----------------------------
def scale_features(train_features, test_features):
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    return train_scaled, test_scaled


# -----------------------------
# 1️⃣ Zero-Shot Evaluation
# -----------------------------
def run_zero_shot(model, subjects):
    train_features, train_labels = load_subjects(subjects, use_seen=True)
    test_features, test_labels = load_subjects(subjects, use_seen=False)
    X_train, X_test = scale_features(train_features, test_features)
    model.fit(X_train, train_labels)
    preds = model.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    print(f"\nZero-shot accuracy: {acc:.3f}")
    return acc


# -----------------------------
# 2️⃣ Leave-One-Subject-Out (LOSO)
# -----------------------------
def run_LOSO(model, subjects):
    accuracies = []
    for test_subj in subjects:
        print(f"\n=== LOSO Fold: Testing on {test_subj} ===")
        train_subj = [s for s in subjects if s != test_subj]

        X_train, y_train = load_subjects(train_subj, use_seen=True)
        X_test, y_test = load_subjects([test_subj], use_seen=True)
        X_train, X_test = scale_features(X_train, X_test)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"{test_subj} accuracy: {acc:.3f}")
        print("Unique labels in train:", np.unique(y_train))
        print("Unique labels in test: ", np.unique(y_test))
        # create hashes of each row to check overlap quickly
        train_hashes = set(map(lambda r: hash(r.tobytes()), X_train))
        test_hashes = set(map(lambda r: hash(r.tobytes()), X_test))
        overlap = len(train_hashes & test_hashes)
        print("Number of identical samples in train ∩ test:", overlap)  # should be 0
        print("X_train", X_train.shape, "y_train", y_train.shape)
        print("X_test ", X_test.shape,  "y_test ", y_test.shape)


    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\nAverage LOSO accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    return accuracies


# -----------------------------
# Run experiments
# -----------------------------
if __name__ == "__main__":
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    model = KNeighborsClassifier(n_neighbors=1)
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
    """
    model = StackingClassifier(
        estimators=[
            ('brain', brain),
            ('image', image),
            ('text', text),

        ],
        final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        passthrough=False)"""

    print("\n==============================")
    print(" 1️⃣  Zero-Shot Evaluation ")
    print("==============================")
    run_zero_shot(model, subjects)

    print("\n==============================")
    print(" 2️⃣  Leave-One-Subject-Out (LOSO) Evaluation ")
    print("==============================")
    #run_LOSO(model, subjects)
