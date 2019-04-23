# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import random as rm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

train = pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')
sample = pd.read_csv('sample.csv')

train.columns


train['center'] = (3750901.5068 <= train['x_exit']) & (train['x_exit'] <= 3770901.5068) & (-19268905.6133 <= train['y_exit']) & (train['y_exit'] <= -19208905.6133)

train['center'].value_counts()

train.groupby('hash').any()

test.hash.value_counts()


# clustering entry points
x_ent = np.linspace(train.x_entry.min(),train.x_entry.max(),10)
y_ent = np.linspace(train.y_entry.min(),train.y_entry.max(),10)

entrances = []
for x in x_ent:
    for y in y_ent:
        entrances.append([x,y])
        
knn = pd.DataFrame(entrances)
knn.columns = ['x','y']

plt.scatter(train.x_entry.head(10000), train.y_entry.head(10000))
plt.scatter(knn.x,knn.y, c='r')


neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(np.array(knn))

clustered_entry = neigh.kneighbors(np.array(pd.DataFrame([train.x_entry, train.y_entry]).T))[1]
clustered_exit = neigh.kneighbors(np.array(pd.DataFrame([train.x_exit, train.y_exit]).T))[1]

train['clustered_entry'] = pd.DataFrame(clustered_entry)[0]
train['clustered_exit'] = pd.DataFrame(clustered_exit)[0]
train.loc[train['center'] == True, 'clustered_exit'] = 999


clustered_points
