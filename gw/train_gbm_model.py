import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from matplotlib import style
style.use("ggplot")
from sklearn import svm
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import time

# load the dataset
image_file='./dataset/gw_%s_images.npy'
labels_file='./dataset/gw_%s_labels.npy'
model_file = './models/gw_gbm.model'
gw_roc_file = './models_images/gw_gbm_roc.png'

RANDOM_SEED = 150914 # from event GW150914, first gravitational wave event detected, :)
SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170

np.random.seed(RANDOM_SEED)

images = np.load(image_file % "train", allow_pickle=True)
labels = np.load(labels_file % "train", allow_pickle=True)
images_validation = np.load(image_file % "validation", allow_pickle=True)
labels_validation = np.load(labels_file % "validation", allow_pickle=True)

images = images.reshape(images.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels = labels.astype('int')
images_validation =images_validation.reshape(images_validation.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels_validation = labels_validation.astype('int')


# # Train the light GBM
# model = LGBMClassifier()
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, images, labels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset
model = LGBMClassifier()

start_time = time.time()

model = model.fit(images, labels)
print("Train Light GBM --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
basic_score = model.score(images_validation, labels_validation)
print("Validation Light GBM --- %s seconds ---" % (time.time() - start_time))

print("Light GBM scikit learn basic score: %0.4f" % basic_score)

# Validating the model and evaluation
start_time = time.time()
scores = cross_validate(model, images_validation, labels_validation, cv=5, scoring=('f1','roc_auc_ovo'), return_train_score=True)
print("Cross Validation Light GBM --- %s seconds ---" % (time.time() - start_time))

cross_score = model.score(images_validation, labels_validation)


print("Light GBM scikit learn cross-val score: %0.4f" % cross_score)
print(scores)

pickle.dump(model, open(model_file, 'wb'))

# calculate the fpr and tpr for all thresholds of the classification

probs = model.predict_proba(images_validation)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(labels_validation, preds)
roc_score = roc_auc_score(labels_validation, preds)
print("ROC score: %s" % roc_score)
roc_auc = metrics.auc(fpr, tpr)

# ploting to a file

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig(gw_roc_file, bbox_inches = 'tight',pad_inches = 0)


# Training report

target_names = [1,0]
pred_labels = model.predict(images_validation)
print(classification_report(labels_validation, pred_labels))