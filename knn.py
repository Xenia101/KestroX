# K-Nearest Neighbor
import load_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier, NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

where = './user'
kestrox_data = load_data.get_data()
X_train, X_test, y_train, y_test = kestrox_data.ks_data(where)

all_data = load_data.get_data()
X, y = all_data.get_X_y(where)

n_neighbors = 3
random_state = 0

dim = len(X[0])
n_classes = len(np.unique(y))

pca = make_pipeline(StandardScaler(),
                    PCA(n_components=2, random_state=random_state))
lda = make_pipeline(StandardScaler(),
                    LinearDiscriminantAnalysis(n_components=2))
nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=2,
                                                   random_state=random_state))

knn = KNeighborsClassifier(n_neighbors=n_neighbors)

dim_reduction_methods = [('PCA', pca), ('LDA', lda), ('NCA', nca)]

for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    
    model.fit(X_train, y_train)
    knn.fit(model.transform(X_train), y_train)
    acc_knn = knn.score(model.transform(X_test), y_test)
    
    X_embedded = model.transform(X)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn))
plt.show()