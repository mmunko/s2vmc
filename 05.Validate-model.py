#!/usr/bin/python3

import numpy as np
import os, sys
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc

clasificator = sys.argv[1]

if not os.path.exists('validation'):
    os.mkdir('validation')

def plotROC(fpr,tpr,AUC,title):
  figure = plt.figure()
  figure.suptitle(title.split('/')[-1][:-4],fontsize=12, fontweight='bold')
  plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' %(AUC))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend(loc="lower right")
  plt.savefig('validation/' + title.split('/')[-1][:-4] + '.pdf')
  plt.show()

model = joblib.load(clasificator)

data = np.load('bin/test_data.npy')
labels = np.load('bin/test_labels.npy')

predicted = model.predict_proba(data)[:,1]
fpr, tpr, tresholds = roc_curve(labels,predicted)
AUC = auc(fpr,tpr)

plotROC(fpr,tpr,AUC,clasificator)
