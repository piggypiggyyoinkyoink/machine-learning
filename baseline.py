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
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, precision_recall_fscore_support, confusion_matrix, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
import mmbra
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize



def load_data(nc=100):
    #Data paths
    data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
    image_model = 'pytorch/cornet_s'
    text_model = 'CLIPText'
    roi = '17channels'

    #Initialise variables
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    train_brain, train_image, train_text, train_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    fine_tune_brain, fine_tune_image, fine_tune_text, fine_tune_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    test_brain, test_image, test_text, test_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    NUM_CLASSES = nc
    SAMPLES_PER_CLASS = 10
    TRAIN_PER_CLASS = 6
    FINE_TUNE_PER_CLASS = 2

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
            test_brain[num].append(brain[start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS:end])
            test_image[num].append(image[start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS:end])
            test_text[num].append(text[start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS:end])
            test_label[num].append(label[start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS:end])

        #FINE TUNING DATA
        for i in range(NUM_CLASSES):
            start = i * SAMPLES_PER_CLASS
            end = start + SAMPLES_PER_CLASS
            fine_tune_brain[num].append(brain[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS])
            fine_tune_image[num].append(image[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS])
            fine_tune_text[num].append(text[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS])
            fine_tune_label[num].append(label[start+TRAIN_PER_CLASS:start+TRAIN_PER_CLASS+FINE_TUNE_PER_CLASS])
        
        num+=1 #important

    #Initialise lists
    train_features = []
    fine_tune_features = []
    test_features = []
    train_labels = []
    fine_tune_labels = []
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
        fine_tune_brain[i] = np.vstack(fine_tune_brain[i])
        fine_tune_image[i] = np.vstack(fine_tune_image[i])
        fine_tune_text[i] = np.vstack(fine_tune_text[i])
        fine_tune_label[i] = np.vstack(fine_tune_label[i])
        #add to final lists
        fine_tune_features.append(np.hstack((fine_tune_brain[i], fine_tune_image[i], fine_tune_text[i])))
        fine_tune_labels.append(fine_tune_label[i].ravel())

        #Test data
        test_brain[i] = np.vstack(test_brain[i])
        test_image[i] = np.vstack(test_image[i])
        test_text[i] = np.vstack(test_text[i])
        test_label[i] = np.vstack(test_label[i])
        #add to final lists
        test_features.append(np.hstack((test_brain[i], test_image[i], test_text[i])))
        test_labels.append(test_label[i].ravel())
    #Return
    return train_features, train_labels, fine_tune_features, fine_tune_labels, test_features, test_labels


def fit_preprocessors(train_features, ac=445):
    # pool all subjects’ TRAIN data
    X = np.vstack(train_features)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

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
        #Preprocess data
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
    return mean




def finetune_model(train_mean, X, y, scaler=None, agglomeration=None):
    #Preprocesing
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
    #Fine-tune with fine tuning data
    clf.fit(X, y)
    #Return fine-tuned model
    return clf



def visualise_shit(out_dir, X_test, y_test, model, scaler=None, agglomeration=None):
    X_test = preprocess_data(X_test, scaler=scaler, agglomeration=agglomeration)
    # fix off-by-one error caused by class labels starting at 1
    classes = np.arange(1, NUM_CLASSES + 1)

    # predicted probabilities (n_samples x n_classes)
    y_score = model.predict_proba(X_test)

    # binarize true labels for multi-class PR calculations (shape n_samples x n_classes)
    y_test_bin = label_binarize(y_test, classes=classes)  # classes must match label universe

    # predicted labels (for confusion matrix & classification report)
    y_pred = model.predict(X_test)

    # ---------- 1) Per-class Precision-Recall curves and AP ----------
    # precision = dict()
    # recall = dict()
    # pr_thresholds = dict()
    avg_prec = dict()

    for i, cls in enumerate(classes):
        # precision[i], recall[i], pr_thresholds[i] = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        avg_prec[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    # Macro-average AP (mean of class APs)
    mean_precision = np.mean(list(avg_prec.values()))

    # Histogram of per-clas average precision (AP)
    ap_table = [(int(classes[i]), float(avg_prec[i])) for i in range(len(classes))]
    ap_table = sorted(ap_table, key=lambda x: x[1], reverse=True)  # high->low
    # plot histogram
    plt.figure(figsize=(8,4))
    aps = [v for _,v in ap_table]
    plt.bar(range(len(aps)), aps)
    plt.xlabel("Classes (sorted by AP)")
    plt.ylabel("Average Precision (AP)")
    plt.title("Per-class Average Precision (sorted)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ap_histogram.png"), dpi=200)
    plt.close()

    # PR curve
    # concat all classes into one binary decision (micro-average)
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    avg_precision = average_precision_score(y_test_bin, y_score, average="micro")
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_macro_micro.png"), dpi=200)
    plt.close()


    # ---------- 3) F1 / precision / recall per class and macro/micro/weighted ----------
    #prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix 
    cm = confusion_matrix(y_test, y_pred, labels=classes)  # shape (C, C)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)  # per-row normalization
    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Proportion')
    plt.xlabel('Predicted class index (1..C)')
    plt.ylabel('True class index (1..C)')
    plt.title('Normalised Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_norm.png"), dpi=200)
    plt.close()


    # Extract top confused pairs from confusion matrix
    K = 20  # K=number of pairs to display
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    flat = []
    for i in range(cm_offdiag.shape[0]):
        for j in range(cm_offdiag.shape[1]):
            if cm_offdiag[i,j] > 0:
                flat.append(((int(classes[i]), int(classes[j])), int(cm_offdiag[i,j])))
    flat_sorted = sorted(flat, key=lambda x: x[1], reverse=True)
    top_confused = flat_sorted[:K]
    #write to text file
    with open(os.path.join(out_dir, "top_confused_pairs.txt"), "w") as f:
        f.write("Top confused class pairs (true -> pred) with counts:\n")
        for (t,p),count in top_confused:
            f.write(f"{t} -> {p} : {count}\n")


    # Overall classification report summary
    report = classification_report(y_test, y_pred, labels=classes, digits=3, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write("Classification report (per-class precision/recall/f1/ support):\n")
        f.write(report)
        f.write("\n\n")
        f.write(f"Macro F1: {f1_macro:.4f}\nMicro F1: {f1_micro:.4f}\nWeighted F1: {f1_weighted:.4f}\n")
        f.write(f"Macro AP (mean AP): {mean_precision:.4f}\nMicro AP: {avg_precision:.4f}\n")



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
NUM_CLASSES = 100 #number up to 1654
NUM_TRAIN_SUBJECTS = 9 #there is no reason to change this, ever
COEFFICIENT_OF_AGGLOMERATION=450 #number up to 1632

test_subj = subjects[TEST_INDEX]
non_test_subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]

#Load data
train_features, train_labels, ft_features, ft_labels, test_features, test_labels = load_data(nc=NUM_CLASSES)
X_test = test_features[TEST_INDEX]
y_test = test_labels[TEST_INDEX]


#Organise data
X_train_arr = [(train_features[j]) for j in range(10) if  j != TEST_INDEX]
y_train_arr = [(train_labels[j]) for j in range(10) if j != TEST_INDEX]
X_ft = ft_features[TEST_INDEX]
y_ft = ft_labels[TEST_INDEX]
#Fit preprocessors for future use
scaler, agglomeration = fit_preprocessors(X_train_arr, ac=COEFFICIENT_OF_AGGLOMERATION)

#Training
mean = train_model(X_train_arr, y_train_arr, num_subjects=NUM_TRAIN_SUBJECTS, scaler=scaler, agglomeration=agglomeration)

#Fine tuning
model = finetune_model(mean, X_ft, y_ft, scaler=scaler, agglomeration=agglomeration)

#Testing
acc = test_model(model, X_test, y_test, scaler=scaler, agglomeration=agglomeration)
#Visualisations
visualise_shit("visualisations_baseline", X_test, y_test, model, scaler=scaler, agglomeration=agglomeration)
print(f"\nTest Subject: {test_subj}")
print(f"Accuracy: {acc:.3f}")

