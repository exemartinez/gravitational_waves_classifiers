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
import time

# load the dataset
image_file='./dataset/gw_%s_images.npy'
labels_file='./dataset/gw_%s_labels.npy'


RANDOM_SEED = 150914 # from event GW150914, first gravitational wave event detected, :)
SHAPE_SIZE_X = 140
SHAPE_SIZE_Y = 170

np.random.seed(RANDOM_SEED)

images = np.load(image_file % "train", allow_pickle=True)
labels = np.load(labels_file % "train", allow_pickle=True)

images = images.reshape(images.shape[0],SHAPE_SIZE_X * SHAPE_SIZE_Y)
labels = labels.astype('int')


from scipy import stats
import time

#we want basic statistics for the whole training set.
print(images.shape)

# Converting the  numpy array to pandas dataframe
def gimmeImagesSetStatistics(images):

    df = pd.DataFrame(data=images[1:,1:],    # values
    index=images[1:,0],    # 1st column as index
    columns=images[0,1:])  # 1st row as the column names

    start_time = time.time()

    summary = df.describe()
    summary = summary.transpose()
    print(summary.shape)

    summary2 = summary.describe()

    print("All images SUMMARY values: \n %s " % summary2)
    print("Dataset Analysis --- %s seconds ---" % (time.time() - start_time))

# Now we got the statistics for the Chirps images, the gravitational waves!
images_gw = images[np.where(labels > 0)[0]]

print("The complete spectrograms training set statistics: ")
gimmeImagesSetStatistics(images)

print("The statictics for the Gravitational Waves in the training set:")
gimmeImagesSetStatistics(images_gw)
