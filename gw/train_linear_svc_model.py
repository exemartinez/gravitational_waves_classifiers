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


# load the dataset
image_file='./dataset/gw_%s_images.npy'
labels_file='./dataset/gw_%s_labels.npy'
model_file = './models/gw_linearsvc.model'
gw_roc_file = './models_images/gw_linear_svc_roc.png'


SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170

images = np.load(image_file % "train", allow_pickle=True)
labels = np.load(labels_file % "train", allow_pickle=True)
images_validation = np.load(image_file % "validation", allow_pickle=True)
labels_validation = np.load(labels_file % "validation", allow_pickle=True)

images = images.reshape(images.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels = labels.astype('int')
images_validation =images_validation.reshape(images_validation.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels_validation = labels_validation.astype('int')

#np.random.shuffle(labels_validation) # just testing that everything is in the proper place suffling the samples.
#np.random.shuffle(labels) 

# Optimization of hyperparameters
# C_range = np.logspace(-2, 10, 13)

# param_grid = dict(C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# grid.fit(images, labels)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

# Train the Linear SVM
model  = svm.SVC(kernel='linear', C =1.0, probability=True, class_weight="balanced", verbose=0)

model = model.fit(images, labels)

basic_score = model.score(images_validation, labels_validation)

print("Linear SVC scikit learn basic score: %0.4f" % basic_score)

# Validating the model and evaluation
scores = cross_validate(model, images_validation, labels_validation, cv=5, scoring=('f1','roc_auc_ovo'), return_train_score=True)

cross_score = model.score(images_validation, labels_validation)

print("Linear SVC scikit learn cross-val score: %0.4f" % cross_score)
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