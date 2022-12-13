import numpy as np
import pickle
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
import pandas as pd

CORRUPTED_CATEGORIES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur"
]

with open('cifar10c_features_sev5.pickle', 'rb') as handle:
    cifar10c_features_sev5 = pickle.load(handle)

with open('cifar10c_features_sev3.pickle', 'rb') as handle:
    cifar10c_features_sev3 = pickle.load(handle)

with open('cifar10c_features_sev1.pickle', 'rb') as handle:
    cifar10c_features_sev1 = pickle.load(handle)

with open('cifar10c_labels.pickle', 'rb') as handle:
    cifar10c_labels = pickle.load(handle)

with open('cifar10_features.pickle', 'rb') as handle:
    cifar10_features = pickle.load(handle)

with open('cifar10_labels.pickle', 'rb') as handle:
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
model = LogisticRegression(max_iter=150)
print("Clean CIFAR10 Accuracy:")
print(np.mean(cross_val_score(model, cifar10_features, cifar10_labels, n_jobs=-1, cv=4)))

all_corrupt_samples1 = np.concatenate([cifar10c_features_sev1[k] for k in cifar10c_features_sev1.keys()], axis=0)
all_corrupt_labels1 = np.concatenate([cifar10c_labels[k] for k in cifar10c_labels.keys()], axis=0)

print("Corrupt S=1 Accuracy:")
print(np.mean(cross_val_score(model, all_corrupt_samples1, all_corrupt_labels1, n_jobs=-1, cv=4)))

all_corrupt_samples3 = np.concatenate([cifar10c_features_sev3[k] for k in cifar10c_features_sev3.keys()], axis=0)
all_corrupt_labels3 = np.concatenate([cifar10c_labels[k] for k in cifar10c_labels.keys()], axis=0)

print("Corrupt S=3 Accuracy:")
print(np.mean(cross_val_score(model, all_corrupt_samples3, all_corrupt_labels3, n_jobs=-1, cv=4)))

all_corrupt_samples5 = np.concatenate([cifar10c_features_sev5[k] for k in cifar10c_features_sev5.keys()], axis=0)
all_corrupt_labels5 = np.concatenate([cifar10c_labels[k] for k in cifar10c_labels.keys()], axis=0)

print("Corrupt S=5 Accuracy:")
print(np.mean(cross_val_score(model, all_corrupt_samples5, all_corrupt_labels5, n_jobs=-1, cv=4)))

# per-corruption accuracy
results = {}
if eval_model: 
    for corruption in CORRUPTED_CATEGORIES:
        f1 = cifar10c_features_sev1[corruption]
        f3 = cifar10c_features_sev3[corruption]
        f5 = cifar10c_features_sev5[corruption]
        labels = cifar10c_labels[corruption]
        f1_acc = np.mean(cross_val_score(model, f1, labels, n_jobs=-1, cv=4))
        f3_acc = np.mean(cross_val_score(model, f3, labels, n_jobs=-1, cv=4))
        f5_acc = np.mean(cross_val_score(model, f5, labels, n_jobs=-1, cv=4))
        results[corruption] = f"{np.round(f1_acc,2), np.round(f3_acc,2), np.round(f5_acc, 2)}"

df = pd.DataFrame([results])
df.to_latex()

# ccgs = {}
# for class_idx in range(10):
#     ccgs[class_idx] = GaussianMixture(n_components=1).fit(cifar10_features[cifar10_labels==class_idx])

# # model_lda = LinearDiscriminantAnalysis(solver="eigen")
# severity=1
# for corruption in CORRUPTED_CATEGORIES:
#     features = cifar10c_features_sev1[corruption]
#     labels = cifar10_labels
#     for class_idx in range(10):
#         ccgs[f"{corruption}_{severity}_{class_idx}"] = GaussianMixture(n_components=1).fit(features[labels==class_idx])
#     # print(corruption, "severity=1", np.mean(cross_val_score(model_lda, features, labels)))

# severity=3
# for corruption in CORRUPTED_CATEGORIES:
#     features = cifar10c_features_sev3[corruption]
#     labels = cifar10_labels
#     for class_idx in range(10):
#         ccgs[f"{corruption}_{severity}_{class_idx}"] = GaussianMixture(n_components=1).fit(features[labels==class_idx])
#     # print(corruption, "severity=3", np.mean(cross_val_score(model_lda, features, labels)))

# severity=5
# for corruption in CORRUPTED_CATEGORIES:
#     features = cifar10c_features_sev5[corruption]
#     labels = cifar10_labels
#     for class_idx in range(10):
#         ccgs[f"{corruption}_{severity}_{class_idx}"] = GaussianMixture(n_components=1).fit(features[labels==class_idx])
    # print(corruption, "severity=5", np.mean(cross_val_score(model_lda, features, labels)))

def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.pinv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.pinv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    return .5 * (tr_term + det_term + quad_term - N) 

# kl_mvn(ccgs[0].means_[0],ccgs[0].covariances_[0],ccgs[1].means_[0],ccgs[1].covariances_[0])

def plot_dim_reduction(reducer, features, labels, title='', ):
    pca_feat = reducer.transform(features)
    sns.scatterplot(x=pca_feat[:,0], y=pca_feat[:,1], hue=labels, palette="Set2")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(title+".png")
    plt.show()

# plot_dim_reduction(lda_viz, cifar10_features, cifar10_labels, title="LDA Features (CIFAR10)")
plot_dim_reduction(pca, cifar10_features, cifar10_labels, title="PCA Features (CIFAR10)")

cifar10_features_c0 = cifar10_features[cifar10_labels == 0]
cifar10_labels_c0 = cifar10_labels[cifar10_labels == 0]

for corruption in CORRUPTED_CATEGORIES:
    corrupt_feat = cifar10c_features_sev1[corruption][cifar10c_labels[corruption] == 0]
    corrupt_label = cifar10c_labels[corruption][cifar10c_labels[corruption] == 0]

    all_feats = np.concatenate([cifar10_features_c0, corrupt_feat], axis=0)
    all_labels = np.concatenate([np.zeros_like(cifar10_labels_c0), np.ones_like(corrupt_label)], axis=0)
    plot_dim_reduction(pca, all_feats, all_labels, title=f"Clean vs. '{corruption}' Corruption")

# plot_dim_reduction(lda_viz, cifar10_features, cifar10_labels, title="LDA Features (CIFAR10)")
# plot_dim_reduction(pca, cifar10_features, cifar10_labels, title="PCA Features (CIFAR10)")