
#First, set the experiments seeds for reproducible results.
RANDOM_SEED = 150914 # from event GW150914, first gravitational wave event detected, :)
from numpy.random import seed
seed(RANDOM_SEED)
from tensorflow import set_random_seed
set_random_seed(RANDOM_SEED)

import numpy as np
import os
import matplotlib.pyplot as plt
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
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import time


#load the set
image_file='./dataset/gw_%s_images.npy'
model_file = './models/gw_gbm.model'
labels_file='./dataset/gw_%s_labels.npy'
gw_img_file='./dataset/gw_images/gw_%s_%s_gbm.png'
gw_roc_file = './models_images/gw_gbm_roc_predict_%s.png'

model = pickle.load(open(model_file, 'rb'))


'''
calculate the fpr and tpr for all thresholds of the classification
'''
def plot_roc_curve(images, labels, dataset):

    labels = labels.astype('int')

    probs = model.predict_proba(images)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    roc_score = roc_auc_score(labels, preds)
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

    plt.savefig(gw_roc_file % dataset, bbox_inches = 'tight',pad_inches = 0)

'''
Saves the predicted images from the given sub-set
'''
def identify_predicted_gw(subdataset):

    images = np.load(image_file % subdataset)
    labels = np.load(labels_file % subdataset, allow_pickle=True)

    SHAPE_SIZE_X = 140
    SHAPE_SIZE_Y = 170

    images = images.reshape(images.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)

    # make prediction
    print("="*60)

    start_time = time.time()
    prediction = model.predict(images)
    print("Predict Light GBM --- %s seconds ---" % (time.time() - start_time))

    print("Amount of observations: %s in set %s" % (images.shape[0],subdataset))
    print( "Amount of Gravitational Waves identified by the model: %s" % str(prediction[prediction > 0 ].shape[0]))
    print( "Amount of real Gravitational Waves: %s" % str(labels[labels > 0 ].shape[0]))

    # print ROC/AUC curve
    plot_roc_curve(images, labels, subdataset)

    # plot all the images that we classified into png files to check if they are actual GWs.
    images_gw = images[np.where(prediction > 0)[0]]
    indexes = np.where(prediction > 0)[0]

    i = 0 
    # Saves the images plotted and identified by the model for manual revision
    for image in images_gw:
        
        plt.imshow(image.reshape(140,170))
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig( gw_img_file % (subdataset,indexes[i]), bbox_inches = 'tight',pad_inches = 0)
        
        print(i)

        i = i + 1
    
    return

identify_predicted_gw("validation")
print("Now the TEST set...")
identify_predicted_gw("test")


