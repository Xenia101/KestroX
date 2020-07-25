# Dynamic Time Warping
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class dtw:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        
    def DTW(self, a, b):   
        an = a.size
        bn = b.size
        pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
        cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
        cumdist[0,0] = 0

        for ai in range(an):
            for bi in range(bn):
                minimum_cost = np.min([cumdist[ai, bi+1],
                                    cumdist[ai+1, bi],
                                    cumdist[ai, bi]])
                cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
        return cumdist[an, bn]
    
    def dtw_train(self):
        parameters = {'n_neighbors':[2, 4, 8]}
        clf = GridSearchCV(KNeighborsClassifier(metric=self.DTW), parameters, cv=3, verbose=1)
        clf.fit(self.X_train, self.y_train)        

        y_pred = clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
    
    def show_data(self):
        print(self.X_train)
        print(self.X_test)
        print(self.y_train)
        print(self.y_test)
