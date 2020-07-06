import matplotlib.pyplot as plt
import keras as k
import pandas as pd
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
import os
import numpy as np


# load the dataset
image_file='./dataset/gw_%s_images.npy'
labels_file='./dataset/gw_%s_labels.npy'
model_file = './models/gw_convnet.model'
gw_convnet_acc = './models_images/gw_convnet_acc.png'
gw_convnet_loss = './models_images/gw_convnet_loss.png'

SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170

images = np.load(image_file % "train", allow_pickle=True)
labels = np.load(labels_file % "train", allow_pickle=True)
images_val = np.load(image_file % "validation", allow_pickle=True)
labels_val = np.load(labels_file % "validation", allow_pickle=True)


#Designing the right model for the classification of the spectrograms
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation="relu", input_shape=(SHAPE_SIZE_X,SHAPE_SIZE_Y,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# train the recurrent neural network

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['binary_accuracy'])
#model.fit(images, labels, epochs=5, batch_size=100)
history = model.fit(images, labels, epochs=30, steps_per_epoch=100, batch_size=None, validation_data=(images_val, labels_val), validation_steps=50)

model.save(model_file)

# Print training images

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['acc']
val_loss = history.history['val_acc']

epochs = range(1, len(acc)+ 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title("Trainig and Validation Accuracy - ConvNets")
plt.legend()

plt.savefig(gw_convnet_acc, bbox_inches = 'tight',pad_inches = 0)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title("Trainig and Validation Loss - ConvNets")
plt.legend()

plt.savefig(gw_convnet_loss, bbox_inches = 'tight',pad_inches = 0)

# load the test set 

images = np.load(image_file % "test")
labels = np.load(labels_file % "test" , allow_pickle=True)

# evaluate performance on the test set
test_loss, test_acc = model.evaluate(images, labels)

print("Convolutional Network model LOSS: %s" % str(test_loss))
print("Convolutional Network model ACCURACY: %s" % str(test_acc))








