#!/usr/bin/python3

import numpy as np
import json

from sklearn import cross_validation, svm, grid_search


settings = json.loads(open('settings/grid-search_.json','r').read())
data = np.load('bin/train_data.npy')
labels = np.load('bin/train_labels.npy')

data = data[labels == 1]

if 'test_size' in settings.keys():
    data, _ = cross_validation.train_test_split(data,test_size=settings['test_size'])
    del settings['test_size']

print('Training sample shape: {}'.format(data.shape))

kernel_params = settings['params']
del settings['params']

estimator = svm.OneClassSVM()
clasificator = grid_search.GridSearchCV(estimator,kernel_params,**settings)
model = clasificator.fit(data)

print('Best params: {}'.format(str(model.best_params_)))
