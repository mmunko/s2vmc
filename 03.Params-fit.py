#!/usr/bin/python3

import numpy as np
import json

from sklearn import cross_validation, svm, grid_search


settings = json.loads(open('settings/grid-search.json','r').read())
data = np.load('bin/train_data.npy')
labels = np.load('bin/train_labels.npy')

if 'test_size' in settings.keys():
    data_labels, _ = cross_validation.train_test_split(np.column_stack([data,labels]),test_size=settings['test_size'])
    data = data_labels[:,:-1]
    labels = data_labels[:,-1]
    data_labels = None
    del settings['test_size']

print('Training sample shape: {}'.format(data.shape))
print('Number of pozitive samples: {}'.format(len(labels[labels == 1])))
print('Number of negative samples: {}'.format(len(labels[labels == -1])))

kernel_params = settings['params']
del settings['params']

estimator = svm.SVC()
clasificator = grid_search.GridSearchCV(estimator,kernel_params,**settings)
model = clasificator.fit(data,labels)

print('Best params: {}'.format(str(model.best_params_)))
