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


# load data
data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
sbj = 'sub-10'
image_model = 'pytorch/cornet_s'
text_model = 'CLIPText'
roi = '17channels'
brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

brain_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
brain_seen = brain_seen[:,:,27:60] # 70ms-400ms
brain_seen = np.reshape(brain_seen, (brain_seen.shape[0], -1))
image_seen = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double')*50.0
text_seen = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double')*2.0
label_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
image_seen = image_seen[:,0:100]

brain_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['data'].astype('double')*2.0
brain_unseen = brain_unseen[:, :, 27:60]
brain_unseen = np.reshape(brain_unseen, (brain_unseen.shape[0], -1))
image_unseen = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))['data'].astype('double')*50.0
text_unseen = sio.loadmat(os.path.join(text_dir_unseen, 'text_feat_test.mat'))['data'].astype('double')*2.0
label_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['class_idx'].T.astype('int')
image_unseen = image_unseen[:, 0:100]


brain_seen = torch.from_numpy(brain_seen)
brain_unseen = torch.from_numpy(brain_unseen)
image_seen = torch.from_numpy(image_seen)
image_unseen = torch.from_numpy(image_unseen)
text_seen = torch.from_numpy(text_seen)
text_unseen = torch.from_numpy(text_unseen)
label_seen = torch.from_numpy(label_seen)
label_unseen = torch.from_numpy(label_unseen)

"""
print('seen_brain_samples=', brain_seen.shape[0], ', seen_brain_features=', brain_seen.shape[1])
print('seen_image_samples=', image_seen.shape[0], ', seen_image_features=', image_seen.shape[1])
print('seen_text_samples=', text_seen.shape[0], ', seen_text_features=', text_seen.shape[1])
print('seen_label=', label_seen.shape)
print('unseen_brain_samples=', brain_unseen.shape[0], ', unseen_brain_features=', brain_unseen.shape[1])
print('unseen_image_samples=', image_unseen.shape[0], ', unseen_image_features=', image_unseen.shape[1])
print('unseen_text_samples=', text_unseen.shape[0], ', unseen_text_features=', text_unseen.shape[1])
print('unseen_label=', label_unseen.shape)
"""

#Use 50 categories
index_seen = np.squeeze(np.where(label_seen < 1001, True, False))
index_unseen = np.squeeze(np.where(label_unseen < 1001, True, False))

#Use 20 categories
#index_seen = np.squeeze(np.where(label_seen < 21, True, False))
#index_unseen = np.squeeze(np.where(label_unseen < 21, True, False))

brain_seen = brain_seen[index_seen, :]
image_seen = image_seen[index_seen, :]
text_seen = text_seen[index_seen, :]
label_seen = label_seen[index_seen]
brain_unseen = brain_unseen[index_unseen, :]
image_unseen = image_unseen[index_unseen, :]
text_unseen = text_unseen[index_unseen, :]
label_unseen = label_unseen[index_unseen]

#The ThingsEEG-Text dataset is mainly designed and used for Zero-Shot type research work, because the independence of its training set and test set
#in categories is very suitable for this task. If it needs to be used for other types of tasks
#(such as general classification or cross-modal learning),
#the data may need to be repartitioned. Therefore, we repartition the dataset to make it better for our task
#Define the number of classes and the number of samples per class
num_classes = 1000
samples_per_class = 10
new_train_brain = []
new_train_image = []
new_train_text = []
new_train_label = []

new_test_brain = []
new_test_image = []
new_test_text = []
new_test_label = []

numTrain = 7

for i in range(num_classes):
    start_idx = i * samples_per_class#The starting index of the current class
    end_idx = start_idx + samples_per_class#The end index of the current class
    #Get the data of the current class
    class_data_brain = brain_seen[start_idx:end_idx, :]
    print(class_data_brain[numTrain:].shape)
    #Divided into training set and test set
    new_train_brain.append(class_data_brain[:numTrain])
    new_test_brain.append(class_data_brain[numTrain:])

    class_data_image = image_seen[start_idx:end_idx, :]

    new_train_image.append(class_data_image[:numTrain])
    new_test_image.append(class_data_image[numTrain:])

    class_data_text = text_seen[start_idx:end_idx, :]

    new_train_text.append(class_data_text[:numTrain])
    new_test_text.append(class_data_text[numTrain:])

    class_data_label = label_seen[start_idx:end_idx, :]

    new_train_label.append(class_data_label[:numTrain])
    new_test_label.append(class_data_label[numTrain:])

train_brain = torch.vstack(new_train_brain)
train_image = torch.vstack(new_train_image)
train_text = torch.vstack(new_train_text)
train_label = torch.vstack(new_train_label)
test_brain = torch.vstack(new_test_brain)
test_image = torch.vstack(new_test_image)
test_text = torch.vstack(new_test_text)
test_label = torch.vstack(new_test_label)
"""
print(train_brain.shape)
print(train_image.shape)
print(train_text.shape)
print(train_label.shape)
print(test_brain.shape)
print(test_image.shape)
print(test_text.shape)
print(test_label.shape)
"""
train_brain_np = train_brain.numpy()
train_image_np = train_image.numpy()
train_text_np = train_text.numpy()
train_label_np = train_label.numpy().ravel()

test_brain_np = test_brain.numpy()
test_image_np = test_image.numpy()
test_text_np = test_text.numpy()
test_label_np = test_label.numpy().ravel()

train_features = np.hstack((train_brain_np, train_image_np, train_text_np))
test_features = np.hstack((test_brain_np, test_image_np, test_text_np))

scaler = preprocessing.StandardScaler().fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)


pca = PCA(n_components=0.95)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)

lda = LinearDiscriminantAnalysis()
train_features_lda = lda.fit_transform(train_features, train_label_np)
test_features_lda = lda.transform(test_features)

train_features_both = pca.fit_transform(train_features_lda)
test_features_both = pca.transform(test_features_lda)

print(train_features.shape)
print(train_features_pca.shape)
print(train_features_lda.shape)
print(train_features_both.shape)

#train_features = train_features_lda
#test_features=test_features_lda

svm_clf = svm.SVC(kernel='linear')
log_clf = LogisticRegression(max_iter=1000)
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf2 = KNeighborsClassifier(n_neighbors=1)
knn_clf3 = KNeighborsClassifier(n_neighbors=2)
knn_clf4 = KNeighborsClassifier(n_neighbors=5)

model = StackingClassifier(
    estimators=[
        ('lr', log_clf),
        ('knn', knn_clf),
        ('knn2', knn_clf2),
        ('knn3', knn_clf3),
        ('knn4', knn_clf4),
        ('svm', svm_clf)
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    passthrough=False
)
#"""
model.fit(train_features, train_label_np)
test_predictions = model.predict(test_features)

accuracy = accuracy_score(test_label_np, test_predictions)
print("Accuracy on test data:", accuracy)
print("Classification report:")
print(classification_report(test_label_np, test_predictions, zero_division=0.0))
#"""