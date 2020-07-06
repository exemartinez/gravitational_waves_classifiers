import pandas as pd
import os
import numpy as np
from numpy import save
from keras.utils import to_categorical

'''
Reads a specific matrix and reformates it to be accepted by Keras Deep learning models.
'''
def __normalize_matrix(images):
    
    formatted_images = np.zeros((images.shape[0],SHAPE_SIZE_X, SHAPE_SIZE_Y, 1))

    for i in range(0,images.shape[0]):
        image = images[i]

        for x in range(0,SHAPE_SIZE_X-1):
            for y in range(0,SHAPE_SIZE_Y-1):
                formatted_images[i][x][y][0] = image[x][y]
                #train_formatted_images[i][x][y][0] = None
        
        print("Transformed image %s of %s." % (str(i+1),str(images.shape[0])), end='\r')

    print("")
    print("Finished")

    return formatted_images

'''
We use this function to extract the due internal data already classified by type;
we do normalize it as well for training and validation.
'''
def __extract_matrix(type, df):
    dataset = df.loc[df["sample_type"] == type]

    images = dataset["png"]
    labels = dataset["label"]

    labels.loc[labels !='Chirp'] = 0 # non-gravitational wave
    labels.loc[labels =='Chirp'] = 1 # gravitational wave

    print("Amount of non-gw #%s for %s." % (str(labels.loc[labels !=1].shape[0]), type))
    print("Amount of gw #%s for %s." % (str(labels.loc[labels ==1].shape[0]), type))

    images = images.to_numpy()
    #labels = to_categorical(labels, num_classes=2)

    return images, labels

'''
Executes the main functionality
'''
def __process_dataset(type, df):

    image_file='./dataset/gw_%s_images.npy'
    labels_file='./dataset/gw_%s_labels.npy'

    images, labels = __extract_matrix(type, df)
    images = __normalize_matrix(images)

    save(image_file % type, images)
    save(labels_file % type, labels)

# loads the dataset
pickle_file='./dataset/gw_consolidated_matrix.pickle'
df = pd.read_pickle(pickle_file)

SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170
MATRIX_SIZE = df.shape[0]

print(df.columns)

# Prepares the dataset with which we will train the model
print("Dataset shape: %s" % str(df.shape))
print("Amount of gravitational waves in the dataset: %s" % str(df.query("label =='Chirp'").shape[0] ))

# Normalizing the datasets
__process_dataset("train", df)
__process_dataset("test", df)
__process_dataset("validation", df)
