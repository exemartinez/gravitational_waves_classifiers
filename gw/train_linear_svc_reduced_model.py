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
model_file = './models/gw_linearsvc_reduced.model'
gw_roc_file = './models_images/gw_linear_svc_reduced_roc.png'

RANDOM_SEED = 150914 # from event GW150914, first gravitational wave event detected, :)
SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170
NO_GW_SAMPLES = 41

np.random.seed(RANDOM_SEED)

images = np.load(image_file % "train", allow_pickle=True)
labels = np.load(labels_file % "train", allow_pickle=True)
images_validation = np.load(image_file % "validation", allow_pickle=True)
labels_validation = np.load(labels_file % "validation", allow_pickle=True)

images = images.reshape(images.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels = labels.astype('int')
images_validation =images_validation.reshape(images_validation.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels_validation = labels_validation.astype('int')

'''
Reduces the dataset balancing its binary classes to 50/50
'''
def reduced_dataset(images, labels):
    # gets all the waves and non waves together.
    images_gw = images[np.where(labels > 0)[0]]
    labels_gw = labels[np.where(labels > 0)]
    images_nogw = images[np.where(labels == 0)[0]]
    labels_nogw = labels[np.where(labels == 0)]

    gw = np.column_stack((images_gw, labels_gw))
    nogw = np.column_stack((images_nogw, labels_nogw))

    #randomize the no-gw samples
    nogw = nogw[np.random.choice(nogw.shape[0], NO_GW_SAMPLES, replace=False), :]

    # stack them all
    all = np.row_stack((gw, nogw))
    np.random.shuffle(all) 

    labels = all[:, (SHAPE_SIZE_X * SHAPE_SIZE_Y)]
    images = all[:, 0:(SHAPE_SIZE_X * SHAPE_SIZE_Y)]

    return images, labels

#reduces the testing dataset
images, labels = reduced_dataset(images, labels)
random_state = np.random.RandomState(RANDOM_SEED)

model  = svm.SVC(kernel='linear', C =0.000075, probability=True, class_weight="balanced", verbose=0, random_state=random_state)
model = model.fit(images, labels)
basic_score = model.score(images_validation, labels_validation)

print("Linear SVC scikit learn accuracy: %0.4f" % basic_score)

pickle.dump(model, open(model_file, 'wb'))

# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(images_validation)
preds = probs[:,1]

roc_score = roc_auc_score(labels_validation, preds)
print("AUC score: %s" % roc_score)

fpr, tpr, threshold = metrics.roc_curve(labels_validation, preds)

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
