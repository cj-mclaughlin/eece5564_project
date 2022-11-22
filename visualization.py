import numpy as np
import pickle
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# with open('cifar10c_features_sev5.pickle', 'rb') as handle:
#     cifar10c_features_sev5 = pickle.load(handle)

# with open('cifar10c_features_sev3.pickle', 'rb') as handle:
#     cifar10c_features_sev3 = pickle.load(handle)

# with open('cifar10c_features_sev1.pickle', 'rb') as handle:
#     cifar10c_features_sev1 = pickle.load(handle)

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
lda = LinearDiscriminantAnalysis(solver="eigen", n_components=2).fit(cifar10_features, cifar10_labels)

def plot_dim_reduction(reducer, features, labels):
    pca_feat = reducer.transform(features)
    sns.scatterplot(x=pca_feat[:,0], y=pca_feat[:,1], hue=labels)
    plt.show()

plot_dim_reduction(lda, cifar10_features, cifar10_labels)