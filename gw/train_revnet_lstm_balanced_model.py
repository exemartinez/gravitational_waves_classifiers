#First, set the experiments seeds for reproducible results.
RANDOM_SEED = 150914 # from event GW150914, first gravitational wave event detected, :)
from numpy.random import seed
seed(RANDOM_SEED)
from tensorflow import set_random_seed
set_random_seed(RANDOM_SEED)

import matplotlib.pyplot as plt
import keras as k
import pandas as pd
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import time
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense

# load the dataset
image_file='./dataset/gw_%s_images.npy'
labels_file='./dataset/gw_%s_labels.npy'
model_file = './models/gw_revnet_lstm.model'
model_opt_file = './models/gw_revnet_lstm_optimal.model'
gw_convnet_acc = './models_images/gw_revnet_lstm_acc.png'
gw_convnet_loss = './models_images/gw_revnet_lstm_loss.png'

SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170
max_features = SHAPE_SIZE_X * SHAPE_SIZE_Y
NO_GW_SAMPLES = 41

images = np.load(image_file % "train", allow_pickle=True)
labels = np.load(labels_file % "train", allow_pickle=True)
images_val = np.load(image_file % "validation", allow_pickle=True)
labels_val = np.load(labels_file % "validation", allow_pickle=True)

images = images.reshape(images.shape[0],max_features)
images_val =  images_val.reshape(images_val.shape[0],max_features)

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


#Designing the right model for the classification of the spectrograms
model = models.Sequential()

model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# train the recurrent neural network

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['binary_accuracy'])
#history = model.fit(images, labels, epochs=5, batch_size=100)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=0, restore_best_weights=True),
    ModelCheckpoint(filepath=model_opt_file, monitor='val_loss', save_best_only=True, verbose=0),
]

start_time = time.time()
history = model.fit(images, labels, epochs=30, batch_size=100, validation_data=(images_val, labels_val), callbacks=callbacks)
print("Train REVNET --- %s seconds ---" % (time.time() - start_time))

model.save(model_file)

# Print training images

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+ 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title("Trainig and Validation Accuracy - ConvNets")
plt.legend()

plt.savefig(gw_convnet_acc, bbox_inches = 'tight',pad_inches = 0)

plt.clf()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title("Trainig and Validation Loss - ConvNets")
plt.legend()

plt.savefig(gw_convnet_loss, bbox_inches = 'tight',pad_inches = 0)

# load the test set 

images = np.load(image_file % "test")
images = images.reshape(images.shape[0],max_features)
labels = np.load(labels_file % "test" , allow_pickle=True)

# evaluate performance on the test set
test_loss, test_acc = model.evaluate(images, labels)

print("Recurrent Network model LOSS: %s" % str(test_loss))
print("Recurrent Network model ACCURACY: %s" % str(test_acc))








