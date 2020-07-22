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
import time


#load the set
image_file='./dataset/gw_%s_images.npy'
model_file = './models/gw_linearsvc.model'
#model_file = './models/gw_linearsvc_reduced_custom.model'
#model_file = './models/gw_linearsvc_reduced.model'
labels_file='./dataset/gw_%s_labels.npy'
gw_img_file='./dataset/gw_images/gw_%s_%s_linearsvc.png'

model = pickle.load(open(model_file, 'rb'))

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
    print("Predict LINEAR SVC --- %s seconds ---" % (time.time() - start_time))

    print("Amount of observations: %s in set %s" % (images.shape[0],subdataset))
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
        
        print(i)

        i = i + 1
    
    return

identify_predicted_gw("validation")
print("Now the TEST set...")
identify_predicted_gw("test")
