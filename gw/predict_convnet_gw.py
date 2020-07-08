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

#load the set
image_file='./dataset/gw_%s_images.npy'
model_file = './models/gw_convnet.model'
labels_file='./dataset/gw_%s_labels.npy'
gw_img_file='./dataset/gw_images/gw_%s_convnet.png'

model = load_model(model_file)

images = np.load(image_file % "validation")
labels = np.load(labels_file % "validation", allow_pickle=True)

# make prediction
print("="*30)
prediction = model.predict_classes(images, verbose=True, batch_size=151)

print( "Amount of Gravitational Waves identified by the model: %s" % str(prediction[prediction > 0 ].shape[0]))
print( "Amount of real Gravitational Waves: %s" % str(labels[labels > 0 ].shape[0]))


# plot all the images that we classified into png files to check if they are actual GWs.
images_gw = images[labels > 0 ]
indexes = np.where(labels >0)

i = 0 
for image in images_gw:
    
    plt.imshow(image.reshape(140,170))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(gw_img_file % indexes[0][i], bbox_inches = 'tight',pad_inches = 0)

    i = i + 1
