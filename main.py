import numpy as np
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import seaborn as sns


CORRUPTED_CATEGORIES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur"
]

with open('resnet18/cifar10c_features_sev5.pickle', 'rb') as handle:
    cifar10c_features_sev5 = pickle.load(handle)

with open('resnet18/cifar10c_features_sev3.pickle', 'rb') as handle:
    cifar10c_features_sev3 = pickle.load(handle)

with open('resnet18/cifar10c_features_sev1.pickle', 'rb') as handle:
    cifar10c_features_sev1 = pickle.load(handle)

with open('resnet18/cifar10c_labels.pickle', 'rb') as handle:
    cifar10c_labels = pickle.load(handle)

with open('resnet18/cifar10_features.pickle', 'rb') as handle:
    cifar10_features = pickle.load(handle)

with open('resnet18/cifar10_labels.pickle', 'rb') as handle:
    cifar10_labels = pickle.load(handle)

class PCA_TSNE():
    def __init__(self, pca, tsne) -> None:
        self.pca = pca
        self.tsne = tsne
    
    def fit(self, features):
        x = self.pca.fit_transform(features)
        return self
    
    def transform(self, features):
        x = self.pca.transform(features)
        x = self.tsne.fit_transform(x)
        return x

pca = PCA(n_components=2, random_state=42).fit(cifar10_features)
# kpca = KernelPCA(n_components=2, kernel="rbf", random_state=42).fit(cifar10_features)
# tsne = PCA_TSNE(PCA(n_components=0.95), tsne=TSNE(learning_rate="auto", random_state=42, init="pca")).fit(cifar10_features)
lda_viz = LinearDiscriminantAnalysis(solver="eigen", n_components=2).fit(cifar10_features, cifar10_labels)

eval_gmm = False
eval_model = True

### FULL DATASET ACCURACY
fog = cifar10c_features_sev3["fog"]
brightness = cifar10c_features_sev3["brightness"]
noise = cifar10c_features_sev3["gaussian_noise"]
    
kf = KFold(n_splits=5)

models = [LogisticRegression, LinearSVC, MLPClassifier]
model_names = ["LogisticRegression", "SVM", "MLP"]
for i, (train_index, test_index) in enumerate(kf.split(cifar10_labels)):
    for model_idx in range(len(models)):
        labels_train, labels_test = cifar10_labels[train_index], cifar10_labels[test_index]
        clean_train, clean_test = cifar10_features[train_index], cifar10_features[test_index]
        fog_train, fog_test = fog[train_index], fog[test_index]
        brightness_train, brightness_test = brightness[train_index], brightness[test_index]
        noise_train, noise_test = noise[train_index], noise[test_index]
        model_clean = models[model_idx]().fit(clean_train, labels_train)
        print(f"{model_names[model_idx]} Clean acc: \
                 clean {model_clean.score(clean_test, labels_test)}, \
                 fog {model_clean.score(fog_test, labels_test)}, \
                 brightness {model_clean.score(brightness_test, labels_test)}, \
                 noise {model_clean.score(noise_test, labels_test)}")
        model_full = models[model_idx]().fit(np.concatenate([clean_train, fog_train, brightness_train, noise_train], axis=0), np.concatenate([labels_train, labels_train, labels_train, labels_train]))
        print(f"{model_names[model_idx]} Conditional acc: \
                 clean {model_full.score(clean_test, labels_test)}, \
                 fog {model_full.score(fog_test, labels_test)}, \
                 brightness {model_full.score(brightness_test, labels_test)}, \
                 noise {model_full.score(noise_test, labels_test)}")

## mid project report - too messy for presentation
# print("Clean CIFAR10 Accuracy:")
# print(np.mean(cross_val_score(model, cifar10_features, cifar10_labels, n_jobs=-1, cv=4)))

# all_corrupt_samples1 = np.concatenate([cifar10c_features_sev1[k] for k in cifar10c_features_sev1.keys()], axis=0)
# all_corrupt_labels1 = np.concatenate([cifar10c_labels[k] for k in cifar10c_labels.keys()], axis=0)

# print("Corrupt S=1 Accuracy:")
# print(np.mean(cross_val_score(model, all_corrupt_samples1, all_corrupt_labels1, n_jobs=-1, cv=4)))

# all_corrupt_samples3 = np.concatenate([cifar10c_features_sev3[k] for k in cifar10c_features_sev3.keys()], axis=0)
# all_corrupt_labels3 = np.concatenate([cifar10c_labels[k] for k in cifar10c_labels.keys()], axis=0)

# print("Corrupt S=3 Accuracy:")
# print(np.mean(cross_val_score(model, all_corrupt_samples3, all_corrupt_labels3, n_jobs=-1, cv=4)))

# all_corrupt_samples5 = np.concatenate([cifar10c_features_sev5[k] for k in cifar10c_features_sev5.keys()], axis=0)
# all_corrupt_labels5 = np.concatenate([cifar10c_labels[k] for k in cifar10c_labels.keys()], axis=0)

# print("Corrupt S=5 Accuracy:")
# print(np.mean(cross_val_score(model, all_corrupt_samples5, all_corrupt_labels5, n_jobs=-1, cv=4)))

## Check per-corruption accuracy
# results = {}
# if eval_model: 
#     for corruption in CORRUPTED_CATEGORIES:
#         f1 = cifar10c_features_sev1[corruption]
#         f3 = cifar10c_features_sev3[corruption]
#         f5 = cifar10c_features_sev5[corruption]
#         labels = cifar10c_labels[corruption]
#         f1_acc = np.mean(cross_val_score(model, f1, labels, n_jobs=-1, cv=4))
#         f3_acc = np.mean(cross_val_score(model, f3, labels, n_jobs=-1, cv=4))
#         f5_acc = np.mean(cross_val_score(model, f5, labels, n_jobs=-1, cv=4))
#         results[corruption] = f"{np.round(f1_acc,2), np.round(f3_acc,2), np.round(f5_acc, 2)}"

# df = pd.DataFrame([results])
# df.to_latex()

## Compare Distance of Data Distributions under Corruptions 
def compute_losses(X, y, model):
    logits = model.predict_log_proba(X)
    logits = torch.from_numpy(logits)
    y = torch.from_numpy(y)
    loss = CrossEntropyLoss(reduction="none")(logits, y).numpy()
    return loss

def fit_ccg(X, y):
    """
    Fit class-conditional Gaussian distribution
    """
    means = []
    covariances = []
    precisions = []
    for c in range(10):
        gaussian = GaussianMixture(n_components=1, n_init=3, random_state=42).fit(X[y==c])
        means.append(gaussian.means_[0])
        covariances.append(gaussian.covariances_[0])
        precisions.append(gaussian.precisions_[0])

    return (means, precisions, covariances)

def bhatta_dist(d1, d2):
    """
    Bhattacharyya distance between two class-conditional multivariate gaussians
    https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    means1, prec1, cov1 = d1
    means2, prec2, cov2 = d2
    dist = 0
    # class conditional gaussian
    for c in range(10):
        m1, p1, c1 = means1[c], prec1[c], cov1[c]
        m2, p2, c2 = means2[c], prec2[c], cov2[c]
        c = (c1 + c2)/2
        p = np.linalg.pinv(c)
        d = (1/8) * ((m1 - m2).T @ p)*(m1 - m2) + (1/2) * np.log(np.linalg.det(c) / np.sqrt(np.linalg.det(c1)*np.linalg.det(c2)))
        dist += d
    return dist

## Correlation of Losses
losses = {}
losses["clean"] = compute_losses(cifar10_features, cifar10_labels, model)
for corruption in CORRUPTED_CATEGORIES:
    features = cifar10c_features_sev3[corruption]
    labels = cifar10_labels
    losses[corruption] = compute_losses(features, labels, model)

df = pd.DataFrame(losses)
ix = df.corr().sort_values('clean', ascending=False).index
df_sorted = df.loc[:, ix]
corr = df_sorted.corr()
plt.figure(figsize=(12, 10))
plt.tight_layout()
sns.heatmap(corr, cmap="Blues", annot=True, linewidth=.5, fmt=".1f")
plt.show()

# compute class conditional gaussians for each category
ccgs = {}
ccgs["clean"] = fit_ccg(cifar10_features, cifar10_labels)
for corruption in CORRUPTED_CATEGORIES:
    features = cifar10c_features_sev3[corruption]
    labels = cifar10_labels
    for class_idx in range(10):
        ccgs[corruption] = fit_ccg(features, labels)

# compute pairwise distances
distance_matrix = np.empty(shape=(len(CORRUPTED_CATEGORIES)+1, len(CORRUPTED_CATEGORIES)+1))
ALL_CATEGORIES = ["clean"] + CORRUPTED_CATEGORIES 
for i in range(len(ALL_CATEGORIES)):
    for j in range(len(ALL_CATEGORIES)):
        distance_matrix[i][j] = bhatta_dist(ccgs[ALL_CATEGORIES[i]], ccgs[ALL_CATEGORIES[j]])

## Scatterplot of Dimensionality Reduced Data
### (Pairwise Clean/Corrupt for each Corruption)
def plot_dim_reduction(reducer, features, labels, title='', ):
    """
    Reduce features to 2 dimensions for scatterplot
    reducer can be fitted PCA or LDA
    """
    pca_feat = reducer.transform(features)
    sns.scatterplot(x=pca_feat[:,0], y=pca_feat[:,1], hue=labels, palette="Set2")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # plt.legend(labels=["Clean", "Corrupted"])
    plt.savefig(title+".png", bbox_inches="tight")
    plt.cla()
    # plt.show()

# plot_dim_reduction(lda_viz, cifar10_features, cifar10_labels, title="LDA Features (CIFAR10)")
# plot_dim_reduction(pca, cifar10_features, cifar10_labels, title="PCA Features (CIFAR10)")

cifar10_features_c0 = cifar10_features[cifar10_labels == 0]
cifar10_labels_c0 = cifar10_labels[cifar10_labels == 0]

for corruption in CORRUPTED_CATEGORIES:
    corrupt_feat = cifar10c_features_sev1[corruption][cifar10c_labels[corruption] == 0]
    corrupt_label = cifar10c_labels[corruption][cifar10c_labels[corruption] == 0]

    all_feats = np.concatenate([cifar10_features_c0, corrupt_feat], axis=0)
    all_labels = np.concatenate([np.zeros_like(cifar10_labels_c0), np.ones_like(corrupt_label)], axis=0)
    text_labels = np.where(all_labels == 0, "Clean", "Corrupted")
    plot_dim_reduction(pca, all_feats, text_labels, title=f"Clean vs. '{corruption}' Corruption")

# show different levels of noise
corrupt1 = cifar10c_features_sev1["gaussian_noise"][cifar10c_labels[corruption] == 1]
c1_label = ["Mild"] * len(corrupt1)
corrupt3 = cifar10c_features_sev3["gaussian_noise"][cifar10c_labels[corruption] == 1]
c3_label = ["Moderate"] * len(corrupt3)
corrupt5 = cifar10c_features_sev5["gaussian_noise"][cifar10c_labels[corruption] == 1]
c5_label = ["High"] * len(corrupt5)
cifar10_features_c1 = cifar10_features[cifar10_labels == 1]
feats = np.concatenate([cifar10_features_c1, corrupt1, corrupt3, corrupt5], axis=0)
clean_label = ["Clean"] * len(corrupt1)
labels = np.concatenate([clean_label, c1_label, c3_label, c5_label], axis=0)
plot_dim_reduction(pca, feats, labels, title=f"Levels of Gaussian Noise Corruption")