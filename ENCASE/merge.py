import os
import pickle
import pandas as pd
import biosppy.signals.ecg as ecg

###RAW DATA
raw_data = pd.read_csv('X_train.csv', index_col='id')

######################################################

###LONG DATA
long_data = raw_data

###QRS DATA
qrs_data = ####!

###SHORT DATA
short_data = []
for wave in range(len(data)):
    signal = raw_data.loc[wave].dropna().to_numpy(dtype='float32')
    r_peaks = ecg.engzee_segmenter(signal, 300)['rpeaks']
    signal_short = []
    for i in range(len(r_peaks)-1):
        signal_short.append(signal[r_peaks[i]:r_peaks[i+1]])
    short_data.append(signal_short)
with open('short_data.pkl', 'wb') as f:
    pickle.dump(short_data, f)

###CENTERWAVE
os.system("python3 centerwave.py")
centerwave = pickle.load("centerwaves.pkl")
centerpoints, centerwaves = centerwave

###EXPANDED DATA
os.system("python3 data_expander.py")
expanded_data = pickle.load("evenet_data.pkl")
exp_data,exp_classes = expanded_data

######################################################

###EXPERT FEATURES

###DNN FEATURES

###CENTERWAVE FEATURES
centerwave_features = []
import features_centerwave
for i in len(centerwaves):
    centerwave_features.append(features_centerwave.GetCenterwaveFeatures(centerwaves[i]))

######################################################

###DNN2?