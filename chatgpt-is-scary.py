import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from itertools import combinations
from tqdm import tqdm
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


# -----------------------------------------------------------
# 1️⃣ Data loader (loads EEG, image, text, labels per subject)
# -----------------------------------------------------------
def load_subjects(subjects, split='train', num_classes=50):
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    all_brain, all_image, all_text, all_label = [], [], [], []

    for sbj in subjects:
        print(f"Loading {sbj} ({split}) ...")
        brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
        image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
        image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
        text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
        text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

        use_seen = split == 'train'

        if True:
            mat_brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))
            brain = mat_brain['data'].astype('double') * 2.0
            label = mat_brain['class_idx'].T.astype('int')
            image = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
            text  = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double') * 2.0
        else:
            mat_brain = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))
            brain = mat_brain['data'].astype('double') * 2.0
            label = mat_brain['class_idx'].T.astype('int')
            image = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))['data'].astype('double') * 50.0
            text  = sio.loadmat(os.path.join(text_dir_unseen, 'text_feat_test.mat'))['data'].astype('double') * 2.0

        brain = brain[:, :, 27:60]      # 70–400 ms window
        brain = np.reshape(brain, (brain.shape[0], -1))
        image = image[:, :100]          # keep first 100 PCA dims

        # filter to first num_classes
        idx = np.squeeze(np.where(label < (num_classes + 1), True, False))
        brain, image, text, label = brain[idx, :], image[idx, :], text[idx, :], label[idx].ravel()

        NUM_TRAIN_PER_CLASS = 1
        NUM_SAMPLES_PER_CLASS = 10
        # (7 train + 3 test per class)
        for c in range(num_classes):
            start = c * NUM_SAMPLES_PER_CLASS
            if split == 'train':
                sel = slice(start, start + NUM_TRAIN_PER_CLASS)
            else:
                sel = slice(start + NUM_TRAIN_PER_CLASS, start + NUM_SAMPLES_PER_CLASS)
            all_brain.append(brain[sel])
            all_image.append(image[sel])
            all_text.append(text[sel])
            all_label.append(label[sel])

    return (np.vstack(all_brain),
            np.vstack(all_image),
            np.vstack(all_text),
            np.hstack(all_label))


# -----------------------------------------------------------
# 2️⃣ Helper functions for LO-shot learning
# -----------------------------------------------------------
def compute_class_prototypes(X, y):
    """Compute per-class means."""
    classes = np.unique(y)
    prototypes = np.array([X[y == c].mean(axis=0) for c in classes])
    return prototypes, classes


def create_pairwise_mixtures(X_proto, y_proto, alpha=0.5):
    """Make mixtures between class prototypes."""
    pairs = list(combinations(range(len(y_proto)), 2))
    mixed = []
    mixed_labels = []
    for (i, j) in pairs:
        mix = alpha * X_proto[i] + (1 - alpha) * X_proto[j]
        mixed.append(mix)
        mixed_labels.append(y_proto[i])  # label of first class
    return np.array(mixed), np.array(mixed_labels)

"""
# -----------------------------------------------------------
# 3️⃣ EEG-space LO-shot (EEG or EEG+Image+Text concatenation)
# -----------------------------------------------------------

def lo_shot_eeg_train_test(X_train, y_train, X_test, y_test, pca_dim=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if pca_dim:
        pca = PCA(n_components=pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    proto, proto_labels = compute_class_prototypes(X_train, y_train)
    Xmix, ymix = create_pairwise_mixtures(proto, proto_labels)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(np.vstack((proto, Xmix)), np.hstack((proto_labels, ymix)))

    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)
"""

# -----------------------------------------------------------
# 4️⃣ EEG → Image → CLIP LO-shot (semantic alignment)
# -----------------------------------------------------------
def lo_shot_eeg_to_clip_with_image(X_train_brain, X_train_img, X_train_text,
                                   y_train, X_test_brain, y_test,
                                   alpha_eeg2img=1.0, alpha_img2clip=1.0, pca_dim=None):

    scaler = StandardScaler()
    X_train_brain = scaler.fit_transform(X_train_brain)
    X_test_brain = scaler.transform(X_test_brain)

    if pca_dim:
        pca = PCA(n_components=pca_dim)
        X_train_brain = pca.fit_transform(X_train_brain)
        X_test_brain = pca.transform(X_test_brain)

    eeg_proto, labels = compute_class_prototypes(X_train_brain, y_train)
    img_proto, _  = compute_class_prototypes(X_train_img, y_train)
    txt_proto, _  = compute_class_prototypes(X_train_text, y_train)

    # regress EEG→Image and Image→CLIP
    reg_eeg2img = Ridge(alpha=alpha_eeg2img).fit(eeg_proto, img_proto)
    reg_img2clip = Ridge(alpha=alpha_img2clip).fit(img_proto, txt_proto)

    eeg2clip_proto = reg_img2clip.predict(reg_eeg2img.predict(eeg_proto))
    Xmix, ymix = create_pairwise_mixtures(eeg2clip_proto, labels)

    """
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(np.vstack((eeg2clip_proto, Xmix)), np.hstack((labels, ymix)))
"""
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', C=10, gamma='scale')
    clf.fit(np.vstack((eeg2clip_proto, Xmix)), np.hstack((labels, ymix)))

    test_clip_pred = reg_img2clip.predict(reg_eeg2img.predict(X_test_brain))
    preds = clf.predict(test_clip_pred)
    return accuracy_score(y_test, preds)


# -----------------------------------------------------------
# 5️⃣ LOSO loop – run all three variants
# -----------------------------------------------------------
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
avgs = [[],[],[]]
for test_subj in subjects:
    print(f"\n=== Leave-One-Subject-Out: Testing on {test_subj} ===")
    train_subj = [s for s in subjects if s != test_subj]

    # load data
    X_train_brain, X_train_img, X_train_text, y_train = load_subjects(train_subj, split='train')
    X_test_brain, X_test_img, X_test_text, y_test    = load_subjects([test_subj], split='test')

    """
    # ---------------------------------------------------
    # ① EEG only
    acc_eeg = lo_shot_eeg_train_test(X_train_brain, y_train, X_test_brain, y_test, pca_dim=200)
    # ② EEG + Image + Text concatenated
    X_train_concat = np.hstack((X_train_brain, X_train_img, X_train_text))
    X_test_concat  = np.hstack((X_test_brain, X_test_img, X_test_text))
    acc_concat = lo_shot_eeg_train_test(X_train_concat, y_train, X_test_concat, y_test, pca_dim=200)
    # ③ EEG → Image → CLIP alignment
    """
    acc_semantic = lo_shot_eeg_to_clip_with_image(
        X_train_brain, X_train_img, X_train_text,
        y_train, X_test_brain, y_test,
        alpha_eeg2img=1.0, alpha_img2clip=1.0, pca_dim=200
    )
    #avgs[0].append(acc_eeg)
    #avgs[1].append(acc_concat)
    avgs[2].append(acc_semantic)
    #print(f"EEG only: {acc_eeg:.4f} | EEG+Img+Text: {acc_concat:.4f} | EEG→Image→CLIP: {acc_semantic:.4f}")
    print(f"Accuracy: {acc_semantic:.4f}")

#average1 = sum(avgs[0])/len(avgs[0])
#average2 = sum(avgs[1])/len(avgs[1])
average3 = sum(avgs[2])/len(avgs[2])
#print(average1,"\n", average2,"\n", average3)
print(f"Overall average: {average3:.4f}")