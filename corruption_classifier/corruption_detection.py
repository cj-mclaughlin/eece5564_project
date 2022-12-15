import numpy as np
import pickle
from mlp import MLP
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

with open('cifar10c_features.pickle', 'rb') as handle:
    cifar10c = pickle.load(handle) 


keyes = list(cifar10c.keys())
x = []
y = []
for key in keyes:
    features = np.array(cifar10c[key])
    if key == 'brightness':
        labels = np.zeros(len(features))
    elif key == 'contrast':
        labels = np.ones(len(features))
    elif key == 'fog':
        labels = np.ones(len(features)) * 2
    elif key == 'gaussian_noise':
        labels = np.ones(len(features)) * 3
    # Append features and labels of every key to x and y
    x.append(features)
    y.append(labels)
x = np.array(x)
y = np.array(y)
x = x.reshape(-1, 512)
y = y.reshape(-1, 1)

idx = np.arange(x.shape[0])
np.random.shuffle(idx)
x = x[idx]
y = y[idx]

best_score = 0
best_model = None
best_hidden = 0
progress = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100)
folds = KFold(n_splits = 10, shuffle = True, random_state = 100)
for hidden in [512, 1024, 2048, 4096]:
    score = 0
    for train_index, val_index in folds.split(x_train, y_train):
        X_train, X_val = x_train[train_index], x_train[val_index]
        Y_train, Y_val = y_train[train_index], y_train[val_index]
        model = MLP(hidden)
        model.fit(X_train, Y_train)
        score += model.score(X_val, Y_val)
    score /= 10
    if score > best_score:
        best_score = score
        best_model = model
        best_hidden = hidden
print(f"MLP Results: Best score: {best_score}, Best hidden: {best_hidden}")
y_pred = np.argmax(best_model.predict(x_test), axis=1)
# normalized confusion matrix
cm = confusion_matrix(y_test.ravel(), y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
x_axis_labels = ['brightness', 'contrast', 'fog', 'gaussian_noise'] # labels for x-axis

sns.heatmap(cm, xticklabels=x_axis_labels, yticklabels=x_axis_labels,annot=True, fmt='.2f')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('MLP Confusion Matrix')
plt.savefig('mlp_cm.png')















