from sklearn.preprocessing import StandardScaler
import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
NUM_CLASSES = 120
SAMPLES_PER_CLASS = 10
VIS_PER_CLASS = 1

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
    label = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
    
    #Idk what this does but it was in the colab code and i aint touching it
    brain = brain[:, :, 20:80]  # 70ms–400ms
    brain = np.reshape(brain, (brain.shape[0], -1))

    #Only get the correct number of classes
    idx = np.squeeze(np.where(label < (NUM_CLASSES+1), True, False))
    brain, label = brain[idx, :], label[idx]
    
    #TRAINING DATA
    for i in range(NUM_CLASSES):
        start = i * SAMPLES_PER_CLASS
        end = start + SAMPLES_PER_CLASS
        train_brain[num].append(brain[start:start+VIS_PER_CLASS])
    num+=1 #important


for i in range(10):
    #Train data
    train_brain[i] = np.vstack(train_brain[i])
    print(np.array(train_brain[i]).shape)
print(np.array(train_brain).shape)
X_all = np.array([train_brain[s] for s in range(10)])

X_flat = X_all.reshape(-1, X_all.shape[-1])
subjects = np.repeat(np.arange(X_all.shape[0]), X_all.shape[1])

# Scale 
scaler = StandardScaler().fit(X_flat)
X_scaled = scaler.transform(X_flat)

# PCA
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_scaled)

# t-SNE
X_tsne = TSNE(
    n_components=2,
    perplexity=30,
    init='pca',
    random_state=42
).fit_transform(X_pca)


plt.figure(figsize=(7, 6))

for s in np.unique(subjects):
    idx = subjects == s
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        s=8,
        alpha=0.6,
        label=f"sub-{s+1:02d}"
    )

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("T-SNE showing EEG feature space per Subject (120 samples)")
plt.legend(markerscale=2, fontsize=8, ncol=2, title="Subject")
plt.tight_layout()
plt.savefig("tsne_subjects.png", dpi=300)
# plt.show()


