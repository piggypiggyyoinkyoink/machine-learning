from sklearn.calibration import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import scipy.io as sio
import numpy as np
from sklearn import cluster
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, precision_recall_fscore_support, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
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



def fit_preprocessors(train_features, f=445):
    # pool training data
    X = np.vstack(train_features)
    #fit scaler and agglomeration
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    agglomeration = cluster.FeatureAgglomeration(n_clusters=f).fit(X)
    return scaler, agglomeration



def preprocess_data(X, scaler, agglomeration=None):
    #apply preprocessors
    X = scaler.transform(X)
    X = agglomeration.transform(X)
    return X



def train_model(X_train_arr, y_train_arr, num_subjects=7, scaler=None, agglomeration=None):
    clfs = []
    #Train LR model M_i on each training subject i
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
    #Preproccess data
    X = preprocess_data(X, scaler=scaler, agglomeration=agglomeration)
    #Fit classifier - tune all the weights
    m = ClassWeightedVotingClassifier(
    estimators=[('model'+str(i), models[i]) for i in range(len(models))],
    alpha=alpha,              
    )
    m.fit(X, y)
    #Return fine-tuned CWVC
    return m



def visualisations(out_dir, X_test, y_test, model, scaler=None, agglomeration=None):
    X_test = preprocess_data(X_test, scaler=scaler, agglomeration=agglomeration)
    # fix off-by-one error caused by class labels starting at 1
    classes = np.arange(1, NUM_CLASSES + 1)
    # predicted probabilities (shape: num_samples x num_classes)
    y_score = model.predict_proba(X_test)
    # binarise labels for multiclass calculations (shape num_samples x num_classes)
    y_test_bin = label_binarize(y_test, classes=classes)  # classes must match label universe
    # predicted labels 
    y_pred = model.predict(X_test)

    # Histogram of per-class average precision (AP)
    avg_prec = dict()
    for i in range(len(classes)):
        avg_prec[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
    ap_table = [(int(classes[i]), float(avg_prec[i])) for i in range(len(classes))]
    ap_table = sorted(ap_table, key=lambda x: x[1], reverse=True)  # sort from high to low
    # plot histogram
    plt.figure(figsize=(8,4))
    aps = [v for _,v in ap_table]
    plt.bar(range(len(aps)), aps)
    plt.xlabel("Classes (sorted by AP)")
    plt.ylabel("Average Precision (AP)")
    plt.title("Per-class Average Precision (sorted)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histogram.png"), dpi=200)
    plt.close()

    # PR curve
    # concat. all classes into one binary decision (micro-average)
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel(), drop_intermediate=True)
    # remove anomalous points from the curve casued by 0/0 = 1 in sklearn 
    a = [(precision[i], recall[i]) for i in range(len(precision))]
    b = [(e[0], e[1]) for e in a if not ((e[1]==0.0 or (e[0] < 0.7 and e[1] < 0.02)) and e[0] !=1.0)]
    precision = [e[0] for e in b]
    recall = [e[1] for e in b]
    # plot PR curve
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=200)
    plt.close()

    # Confusion matrix 
    cm = confusion_matrix(y_test, y_pred, labels=classes)  # shape (C, C)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)  # normalise each row
    # plot
    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Proportion')
    plt.xlabel('Predicted class index (1..C)')
    plt.ylabel('True class index (1..C)')
    plt.title('Normalised Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


    #Extract all confused pairs (inaccurate predicitoons) from confusion matrix
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat = []
    for i in range(cm_copy.shape[0]):
        for j in range(cm_copy.shape[1]):
            if cm_copy[i,j] > 0:
                flat.append(((int(classes[i]), int(classes[j])), int(cm_copy[i,j])))
    flat_sorted = sorted(flat, key=lambda x: x[1], reverse=True)
    top_confused = flat_sorted
    # write to text file
    with open(os.path.join(out_dir, "confused_pairs.txt"), "w") as f:
        for (t,p),count in top_confused:
            f.write(f"{t} -> {p} : {count}\n")


    #Overall classification report summary
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    micro_precision = average_precision_score(y_test_bin, y_score, average="micro")
    macro_precision = average_precision_score(y_test_bin, y_score, average="macro")

    report = classification_report(y_test, y_pred, labels=classes, digits=3, zero_division=0)
    # write to text file
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\n\nMacro F1: {f1_macro:.4f}\nMicro F1: {f1_micro:.4f}\n")
        f.write(f"Macro AP (mean AP): {macro_precision:.4f}\nMicro AP: {micro_precision:.4f}\n")



def test_model(model, X_test, y_test, scaler=None, agglomeration=None):
    #Preprocess data
    X_test = preprocess_data(X_test, scaler=scaler, agglomeration=agglomeration)
    #Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    f1_macro = f1_score(y_test, y_pred, average='macro',zero_division=0)
    print(f"Test F1 (macro): {f1_macro:.3f}")
    f1_micro = f1_score(y_test, y_pred, average='micro',zero_division=0)
    print(f"Test F1 (micro): {f1_micro:.3f}")
    prec_macro = precision_score(y_test, y_pred, average='macro',zero_division=0)
    print(f"Test Precision (macro): {prec_macro:.3f}")
    rec_macro = recall_score(y_test, y_pred, average='macro',zero_division=0)
    print(f"Test Recall (macro): {rec_macro:.3f}")
    prec_micro = precision_score(y_test, y_pred, average='micro',zero_division=0)
    print(f"Test Precision (micro): {prec_micro:.3f}")
    rec_micro = recall_score(y_test, y_pred, average='micro',zero_division=0)
    print(f"Test Recall (micro): {rec_micro:.3f}")




#PROGRAM START

#Initialise variables
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
TEST_INDEX = 0 #0-9 - which subject to use as test
NUM_CLASSES = 1654 #number up to 1654
NUM_TRAIN_SUBJECTS = 9 #there is no reason to change this, ever
COEFFICIENT_OF_AGGLOMERATION=565  #hyperparameter F for FeatureAgglomeration
ALPHA=1.0  #hyperparameter alpha for CWVC
test_subj = subjects[TEST_INDEX]
non_test_subjects = subjects[0:TEST_INDEX] + subjects[TEST_INDEX+1:]


#Load data
train_features, train_labels, fine_tune_features, fine_tune_labels, test_features, test_labels = load_data(nc=NUM_CLASSES)

#Organise data
X_train_arr = [(train_features[j]) for j in range(10) if j != TEST_INDEX]
y_train_arr = [(train_labels[j]) for j in range(10) if j != TEST_INDEX]
X_ft = fine_tune_features[TEST_INDEX]
y_ft = fine_tune_labels[TEST_INDEX]
X_test = test_features[TEST_INDEX]
y_test = test_labels[TEST_INDEX]

#Fit preprocessors for future use
scaler, agglomeration = fit_preprocessors(X_train_arr, f=COEFFICIENT_OF_AGGLOMERATION)

#Training
modelsList = train_model(X_train_arr, y_train_arr, num_subjects=NUM_TRAIN_SUBJECTS, scaler=scaler, agglomeration=agglomeration)
    
#Fine tuning
model = finetune_model(modelsList, X_ft, y_ft, scaler=scaler, agglomeration=agglomeration, alpha=ALPHA)

#Testing
test_model(model, X_test, y_test, scaler=scaler, agglomeration=agglomeration)

#Visualisations
visualisations("visualisations_improved", X_test, y_test, model, scaler=scaler, agglomeration=agglomeration)

print(f"\nTest Subject: {test_subj}")

