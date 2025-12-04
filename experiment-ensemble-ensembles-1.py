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
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    train_brain, train_image, train_text, train_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    valid_brain, valid_image, valid_text, valid_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    test_brain, test_image, test_text, test_label = [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)], [[] for i in range(10)]
    NUM_CLASSES = nc
    SAMPLES_PER_CLASS = 10
    TRAIN_PER_CLASS = 5
    VALIDATION_PER_CLASS = 3
    TEST_PER_CLASS = 2

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
        brain = np.reshape(brain, (brain.shape[0], -1))
        image = image[:, :100]
        print(brain.shape, image.shape, text.shape, label.shape)
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
    #Return
    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


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
    weights = []

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
        weights.append(clf.coef_)
        clfs.append(clf)
    
    #Average weights across subjects
    weights = np.stack(weights)  
    mean = np.mean(weights, axis=0)
    #Return mean weight to be used for fine tuning
    return mean, clfs



def finetune_model(train_mean, models, X, y, scaler=None, agglomeration=None):
    X = preprocess_data(X, scaler=scaler, agglomeration=agglomeration)

    m = ClassWeightedVotingClassifier(
    estimators=[('model'+str(i), models[i]) for i in range(len(models))],
    alpha=1.0,              
    weighted=True
    )
    m.fit(X, y)
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
    #Return fine-tuned models
    return m, clf




def test_model(model, X_test, y_test, scaler=None, agglomeration=None):
    #Normalise and Agglomerate
    X_test = preprocess_data(X_test, scaler=scaler, agglomeration=agglomeration)
    #Evaluation
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)




def ensemble_train(models, X, y, scaler, agglomeration):
    #Scale data
    X = preprocess_data(X, scaler=scaler, agglomeration=agglomeration)


    #Fit ensemble model
    m = ClassWeightedVotingClassifier(
    estimators=[('model'+str(i), models[i]) for i in range(len(models))],
    alpha=1.0,
    weighted=False
    )
    

    m.fit(X, y)

    #Return the model, scaler and agglomeration for testing purposes
    return m




#PROGRAM START

#Initialise variables
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
TEST_INDEX = 0 #0-9 - which subject to use as test
NUM_CLASSES = 100 #number up to 1654
NUM_TRAIN_SUBJECTS = 7 #there is no reason to change this, ever
COEFFICIENT_OF_AGGLOMERATION=450 #number up to sum(BAC+IAC+TAC)

test_subj = subjects[TEST_INDEX]
non_test_subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]
results = []
models = []

#Load data
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_data(nc=NUM_CLASSES)
X_test = test_features[TEST_INDEX]
y_test = test_labels[TEST_INDEX]
scaler, agglomeration = fit_preprocessors(train_features, ac=COEFFICIENT_OF_AGGLOMERATION)
for i in range(len(non_test_subjects)):
    #Organise data
    val_subj = non_test_subjects[i]
    train_subjects_index = [subjects.index(s) for s in non_test_subjects if s != val_subj]
    X_train_arr = [(train_features[j]) for j in train_subjects_index]
    y_train_arr = [(train_labels[j]) for j in train_subjects_index]
    X_val = valid_features[i]
    y_val = valid_labels[i]

    #Training
    train_mean, modelsList = train_model(X_train_arr, y_train_arr, num_subjects=NUM_TRAIN_SUBJECTS, scaler=scaler, agglomeration=agglomeration)
    
    #Fine tuning
    model, clf = finetune_model(train_mean, modelsList, X_val, y_val, scaler=scaler, agglomeration=agglomeration)
    #Testing
    acc = test_model(model, X_test, y_test, scaler=scaler, agglomeration=agglomeration)
    results.append(acc)
    models.append(model)
    models.append(clf)
    print(f"Accuracy using {val_subj} as validation subject: {acc:.3f}")
#Print average accuracy across all validation subjects    
print("AVERAGE ACCURACY:", np.mean(results))

#Train ensemble model
print("AVENGERS: ENSEMBLE")
X_train = np.vstack(X_train_arr)
y_train = np.hstack(y_train_arr)

ens_X_train, ens_y_train = X_train, y_train

m = ensemble_train(models, ens_X_train, ens_y_train, scaler, agglomeration)

#Test ensemble model
esaccuracy = test_model(m, X_test, y_test, scaler=scaler, agglomeration=agglomeration)
print("ENSEMBLE ACCURACY:", esaccuracy) 

## Accuracy stats (15/11/25)
#1654 classes: 0.6744256348246674 with ensemble, 0.6367 without 
#500 classes: 0.802 with ensemble, 0.753 without
#100 classes: 0.915 with ensemble, 0.883 without

#0.607 weighted = false
#0.612 weighted = true
#0.615 with LR and VC
#0.658 with LR and VC, weighted false for ens



###  CHATGPT VISUALISATIONS









# === ALL DIAGNOSTICS & PLOTS (PR curves, AP table, confusion, F1, dashboard) ===

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report, f1_score
)
import itertools
import os

# ---------- Helpers ----------
def get_probas(model, X_raw, scaler, agglomeration):
    """Return predicted probabilities shape (n_samples, n_classes)."""
    Xp = preprocess_data(X_raw, scaler=scaler, agglomeration=agglomeration)
    # Some sklearn VotingClassifier might not support predict_proba if any base doesn't;
    # fall back to decision_function -> apply softmax for multi-class if necessary.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xp)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(Xp)  # shape (n_samples, n_classes)
        # convert scores -> probabilities via softmax
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs
    else:
        raise RuntimeError("Model has neither predict_proba nor decision_function.")

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# ---------- Compute basic stuff ----------
# y_test expected as 1..NUM_CLASSES labels in your pipeline. Convert consistent classes array:
classes = np.arange(1, NUM_CLASSES + 1)

# predicted probabilities (n_samples x n_classes)
y_score = get_probas(m, X_test, scaler=scaler, agglomeration=agglomeration)

# binarize true labels for multi-class PR calculations (shape n_samples x n_classes)
y_test_bin = label_binarize(y_test, classes=classes)  # classes must match label universe

# predicted labels (for confusion matrix & classification report)
y_pred = m.predict(preprocess_data(X_test, scaler=scaler, agglomeration=agglomeration))

# ---------- 1) Per-class Precision-Recall curves and AP ----------
precision = dict()
recall = dict()
pr_thresholds = dict()
avg_prec = dict()

for i, cls in enumerate(classes):
    precision[i], recall[i], pr_thresholds[i] = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    avg_prec[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# Macro-average AP (mean of class APs)
avg_prec_macro = np.mean(list(avg_prec.values()))

# Save per-class AP table (sorted)
ap_table = [(int(classes[i]), float(avg_prec[i])) for i in range(len(classes))]
ap_table_sorted = sorted(ap_table, key=lambda x: x[1], reverse=True)  # high->low

# ---------- 2) Micro-averaged PR curve ----------
# concat all classes into one binary decision (micro-average)
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
avg_prec["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

# ---------- 3) F1 / precision / recall per class and macro/micro/weighted ----------
prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# ---------- 4) Confusion matrix ----------
cm = confusion_matrix(y_test, y_pred, labels=classes)  # shape (C, C)
cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)  # per-row normalization

# For readability, extract top confused pairs (off-diagonal)
cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)
# get top K confused pairs
K = 20
flat = []
for i in range(cm_offdiag.shape[0]):
    for j in range(cm_offdiag.shape[1]):
        if cm_offdiag[i,j] > 0:
            flat.append(((int(classes[i]), int(classes[j])), int(cm_offdiag[i,j])))
flat_sorted = sorted(flat, key=lambda x: x[1], reverse=True)
top_confused = flat_sorted[:K]

# ---------- 5) Plots & outputs ----------
out_dir = "visualisations_experiment"
ensure_dir(out_dir)

# 5A: Macro PR curve (interpolated mean over classes) + micro PR
plt.figure(figsize=(6,5))
# Plot micro
plt.plot(recall["micro"], precision["micro"], label=f"micro (AP={avg_prec['micro']:.3f})")
# Compute macro-average curve by interpolating each class PR to a common recall grid
# Create a recall grid
all_recall_points = np.linspace(0, 1, 200)
mean_precision_interp = np.zeros_like(all_recall_points)
for i in range(len(classes)):
    # precision[i] is defined at recall[i] points sorted descending; we interpolate
    # ensure monotonic recall for interpolation
    rp = recall[i]
    pp = precision[i]
    # rp is increasing from 0->1, pp corresponding; safe to interpolate
    mean_precision_interp += np.interp(all_recall_points, rp[::-1], pp[::-1])
mean_precision_interp /= len(classes)
#plt.plot(all_recall_points, mean_precision_interp, label=f"macro (mean AP={avg_prec_macro:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Micro Precision-Recall")
plt.legend(loc="best")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pr_macro_micro.png"), dpi=200)
plt.close()

# 5B: Per-class AP histogram
plt.figure(figsize=(8,4))
aps = [v for _,v in ap_table_sorted]
plt.bar(range(len(aps)), aps)
plt.xlabel("Classes (sorted by AP)")
plt.ylabel("Average Precision (AP)")
plt.title("Per-class Average Precision (sorted)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ap_histogram.png"), dpi=200)
plt.close()

# 5C: Plot PR curves for top-k best and worst classes
top_k = 5
best_classes = ap_table_sorted[:top_k]
worst_classes = ap_table_sorted[-top_k:]

plt.figure(figsize=(10,6))
for cls_id, ap in best_classes:
    idx = int(np.where(classes == cls_id)[0])
    plt.plot(recall[idx], precision[idx], label=f"Class {cls_id} (AP={ap:.3f})")
for cls_id, ap in worst_classes:
    idx = int(np.where(classes == cls_id)[0])
    plt.plot(recall[idx], precision[idx], linestyle='--', label=f"Class {cls_id} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curves — Best (solid) and Worst (dashed) classes")
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pr_best_worst.png"), dpi=200)
plt.close()

# 5D: Confusion matrix heatmap (normalized by true class counts)
plt.figure(figsize=(8,6))
plt.imshow(cm_norm, aspect='auto', interpolation='nearest')
plt.colorbar(label='Proportion')
plt.xlabel('Predicted class index (1..C)')
plt.ylabel('True class index (1..C)')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix_norm.png"), dpi=200)
plt.close()

# 5E: Top-K confusion pairs table -> save to txt
with open(os.path.join(out_dir, "top_confused_pairs.txt"), "w") as f:
    f.write("Top confused class pairs (true -> pred) with counts:\n")
    for (t,p),count in top_confused:
        f.write(f"{t} -> {p} : {count}\n")

# 5F: Classification report + F1 summary -> save
report = classification_report(y_test, y_pred, labels=classes, digits=3, zero_division=0)
with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
    f.write("Classification report (per-class precision/recall/f1/ support):\n")
    f.write(report)
    f.write("\n\n")
    f.write(f"Macro F1: {f1_macro:.4f}\nMicro F1: {f1_micro:.4f}\nWeighted F1: {f1_weighted:.4f}\n")
    f.write(f"Macro AP (mean AP): {avg_prec_macro:.4f}\nMicro AP: {avg_prec['micro']:.4f}\n")

# 5G: Save AP table CSV (class, AP)
import csv
with open(os.path.join(out_dir, "ap_table.csv"), "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["class", "average_precision"])
    for c, ap in ap_table_sorted:
        writer.writerow([c, ap])

# 5H: Print a human-readable summary to console
print("=== Diagnostics summary ===")
print(f"Macro AP (mean AP)  : {avg_prec_macro:.4f}")
print(f"Micro AP            : {avg_prec['micro']:.4f}")
print(f"Macro F1            : {f1_macro:.4f}")
print(f"Micro F1            : {f1_micro:.4f}")
print(f"Weighted F1         : {f1_weighted:.4f}")
print()
print("Top 10 classes by AP:")
for c, ap in ap_table_sorted[:10]:
    print(f"  Class {c}: AP={ap:.4f}")
print()
print("Worst 10 classes by AP:")
for c, ap in ap_table_sorted[-10:]:
    print(f"  Class {c}: AP={ap:.4f}")
print()
print(f"Top confused pairs saved to: {os.path.join(out_dir, 'top_confused_pairs.txt')}")
print(f"Plots & tables saved to folder: {out_dir}")
print("Classification report saved to:", os.path.join(out_dir, "classification_report.txt"))
