
#First, set the experiments seeds for reproducible results.
RANDOM_SEED = 150914 # from event GW150914, first gravitational wave event detected, :)
from numpy.random import seed
seed(RANDOM_SEED)
from tensorflow import set_random_seed
set_random_seed(RANDOM_SEED)

import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.models import load_model
import os
import numpy as np
import matplotlib.rcsetup as rcsetup
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import time

#load the set
image_file='./dataset/gw_%s_images.npy'
labels_file='./dataset/gw_%s_labels.npy'
model_file = './models/gw_revnet_gru_dropout.model'
model_opt_file = './models/gw_revnet_gru_optimal_dropout.model'

image_file='./dataset/gw_%s_images.npy'

gw_img_file='./dataset/gw_images/gw_%s_%s_revnet_gru_dropout.png'
gw_roc_file = './models_images/gw_revnet_gru_dropout_%s_roc.png'

SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170
max_features = SHAPE_SIZE_X * SHAPE_SIZE_Y

model = load_model(model_file)

def plot_metrics(model,images,labels,dataset):
    # calculate the fpr and tpr for all thresholds of the classification
    labels = labels.astype("int32")
    preds = model.predict_proba(images).ravel()

    fpr, tpr, threshold = roc_curve(labels, preds)
    roc_score = roc_auc_score(labels, preds, average = 'weighted')
    print("ROC score: %s" % repr(roc_score))
    roc_auc = auc(fpr, tpr)
    print("AUC score: %s" % repr(roc_auc))

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

    # Training report

    target_names = [1,0]
    pred_labels = model.predict_classes(images)
    print(classification_report(labels, pred_labels))

'''
Saves the predicted images from the given sub-set
'''
def identify_predicted_gw(subdataset):

    images = np.load(image_file % subdataset)
    images = images.reshape(images.shape[0],max_features)

    labels = np.load(labels_file % subdataset, allow_pickle=True)

    # make prediction
    print("="*30)

    start_time = time.time()
    prediction = model.predict_classes(images, verbose=True, batch_size=151)
    print("Predict REVNET --- %s seconds ---" % (time.time() - start_time))

    print(" Recursive Neural Network")
    print( "Amount of Gravitational Waves identified by the model: %s" % str(prediction[prediction > 0 ].shape[0]))
    print( "Amount of real Gravitational Waves: %s" % str(labels[labels > 0 ].shape[0]))


    # plot all the images that we classified into png files to check if they are actual GWs.

    images_gw = images[np.where(prediction > 0)[0]]
    indexes = np.where(prediction > 0)[0]

    i = 0 
    for image in images_gw:
        
        plt.imshow(image.reshape(140,170))
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig( gw_img_file % (subdataset,indexes[i]), bbox_inches = 'tight',pad_inches = 0)

        i = i + 1
    
    plot_metrics(model, images, labels,subdataset)

identify_predicted_gw("validation")
identify_predicted_gw("test")