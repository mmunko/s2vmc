#!/usr/bin/python3

import numpy as np
import os, json, datetime
from sklearn.externals import joblib
from sklearn import cross_validation, svm, grid_search


settings = json.loads(open('settings/model-train.json','r').read())
data = np.load('bin/train_data.npy')
labels = np.load('bin/train_labels.npy')

model_name = settings['model_name']
del settings['model_name']

if 'test_size' in settings.keys():
    data_labels, _ = cross_validation.train_test_split(np.column_stack([data,labels]),test_size=settings['test_size'])
    data = data_labels[:,:-1]
    labels = data_labels[:,-1]
    data_labels = None
    del settings['test_size']

print('Training sample shape: {}'.format(data.shape))
print('Number of pozitive samples: {}'.format(len(labels[labels == 1])))
print('Number of negative samples: {}'.format(len(labels[labels == -1])))

# RETRAIN MODEL WITH BEST PARAMS
clasificator = svm.SVC(**settings)
clasificator.fit(data, labels)

if not os.path.exists('results'):
    os.mkdir('results')

joblib.dump(clasificator, os.path.join('results','{}-{}.pkl'.format(model_name,datetime.datetime.now().strftime("%y%m%d-%H%M%S"))))

print()
print('Done.')
