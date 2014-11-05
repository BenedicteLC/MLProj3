#read PCA data from npy file and save them into mat file
import numpy as np
import scipy.io sio

trainXmle = np.load('trainX.npy')
sio.savemat('trainXmle.mat', {'trainXmle':trainXmle})

testXmle = np.load('testX.npy')
sio.savemat('testXmle.mat', {'testXmle':testXmle})
