from scipy.io import loadmat
import numpy as np
 
import random

def load_data(train_data):
    data = loadmat(train_data)
    trajs = data['trajs']
    label = data['label']
    #name  = data['name']
    #namelist = name.tolist()
    time1 = trajs[0]['time']
    label = np.squeeze(label)
    sigsets =[]
    for index in range(trajs.size):
        signal = trajs[index]['X'][0]
        sigsets.append(signal[0])
    return sigsets, time1[0][0], label
