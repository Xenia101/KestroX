from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class get_data:
    def __init__(self):
        self.file_list = list()
        self.l = 0
        self.X = list()
        self.y = list()
        
        self.test_size = 0.2
        self.random_state = 42
        
    def f_search(self, dirname):
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            self.file_list.append(full_path)

    def concat(self, arr):
        if self.l == 0: self.X = arr
        else: self.X = np.vstack((self.X, arr))

    def define_X_y(self):
        mlb = MultiLabelBinarizer()
        
        for x in self.file_list:
            data = pd.read_csv(x)
            data = mlb.fit_transform(data)
            self.concat(data)
        
            for label in range(data.shape[0]):
                self.y.append(self.l)
            self.l+=1
        self.y = np.array(self.y)
    
    def suffle_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = self.test_size,
                                                            random_state = self.random_state) 
        return X_train, X_test, y_train, y_test
    
    def get_X_y(self, dirname):
        self.f_search(dirname)
        self.define_X_y()
        return self.X, self.y
    
    def ks_data(self, dirname):
        self.f_search(dirname)
        self.define_X_y()
        return self.suffle_train_test(self.X, self.y)    
        
    def f_list_show(self):
        print(self.file_list)
        
    def print_dataset(self):
        print(self.X)
        print(self.y)