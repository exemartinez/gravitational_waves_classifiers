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


# load the dataset
image_file='./dataset/gw_%s_images.npy'
labels_file='./dataset/gw_%s_labels.npy'
model_file = './models/gw_linearsvc.model'
gw_roc_file = './models_images/gw_linear_svc_roc.png'


SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170

images = np.load(image_file % "train", allow_pickle=True)
labels = np.load(labels_file % "train", allow_pickle=True)
images_test = np.load(image_file % "test", allow_pickle=True)
labels_test = np.load(labels_file % "test", allow_pickle=True)

images = images.reshape(images.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels = labels.astype('int')
images_test =images_test.reshape(images_test.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels_test = labels_test.astype('int')

# Train the Linear SVM
linearSVC  = svm.SVC(kernel='linear', C = 0.5, probability=True)

model = linearSVC.fit(images, labels)

pickle.dump(model, open(model_file, 'wb'))

# Validating the model and evaluation
scores = cross_validate(model, images_test, labels_test, cv=5, scoring=('accuracy','neg_log_loss','roc_auc'), return_train_score=True)

print(scores)

# calculate the fpr and tpr for all thresholds of the classification

probs = model.predict_proba(images_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(labels_test, preds)
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



