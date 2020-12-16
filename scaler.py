import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def min_max_scaler(x):
    # Scales values between 0 and 1 or -1 and 1
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    return x

def min_max_scaler_train_data(x):
    # Scales values between 0 and 1 or -1 and 1
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    return x

def standard_scaler(x):
    # Scales values in a Normal Distribution with mean of 0 and standard deviation of 1
    ''' Insert here code for StandardScaler '''
    scaler = StandardScaler().fit(x)
    rescaledX = scaler.transform(x)
    np.set_printoptions(precision=3)
    #print(rescaledX[0:5,:1])
    return rescaledX
