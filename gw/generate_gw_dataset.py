'''
Author: Ezequiel H. Martinez
Processes the Gravity Spy dataset with all the spectrograms pre classfied and tags them appropiately.
'''

import gc
import numpy as np
import h5py
import os
import pandas as pd

#load up the hd5f dataset provided by Gravity Spy.
hd5f_file='./dataset/trainingsetv1d1.h5'
pickle_file='./dataset/gw_consolidated_matrix.pickle'

hf = h5py.File(hd5f_file, 'r')

#first, we get the dataframe in pandas with all the headers for the data.
csv = './dataset/trainingset_v1d1_metadata.csv'

gwdf=pd.read_csv(csv, sep=',',header=0)

gvtyid = 21
labelid = 22
sampleid=23

gwdf['png'] = ''

print("The shape of the matrix to be processed is %s" % str(gwdf.shape))

for index, record in gwdf.iterrows():

    gravityspy_id= record[gvtyid]
    label= record[labelid]
    sample_type= record[sampleid]
    
    png = np.array(hf[label][sample_type][gravityspy_id]['0.5.png'][0])
    #png = np.reshape(png,23800).tolist()  #we place the whole image in just one dimension array. 140x170 (remember this)
    
    gwdf.at[index, 'png'] = png

    print("Record #%s processed" % str(index))

print("Main procedures finished. Saving block...")


# save the dataset here
gwdf.to_pickle(pickle_file)

gwdf = None
sdf = None

gc.collect()
