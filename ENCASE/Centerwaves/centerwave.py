import numpy as np
import pandas as pd
from collections import Counter
import biosppy.signals.ecg as ecg
from sklearn.cluster import SpectralClustering


def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix


def calculateCenter():
    for i in range(len(short_data)):
        for j in range(len(short_data)):
            #print(i,j)
            if i==j:
                continue
            dist_matrix[i,j] = dtw(short_data[i], short_data[j])[-1][-1]

    if len(short_data)==1:
        centerpoints.append(0)
        centerwaves.append(short_data[0])
        return
    if len(short_data)<1:
        return 
    clustering = SpectralClustering(n_clusters=min(3, len(short_data)), affinity="precomputed")
    clustering.fit(dist_matrix)

    common_label = Counter(clustering.labels_).most_common(1)[0][0]
    assignments = clustering.labels_
    min_avg_distance = np.Infinity
    for point_index in np.where(assignments == common_label)[0]:
        avg_distance = np.mean(dist_matrix[point_index][assignments == common_label])
        
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            center_point = point_index

    centerpoints.append(center_point)
    centerwaves.append(short_data[center_point])


data = pd.read_csv('X_train.csv', index_col='id')
centerpoints = []
centerwaves = []

for wave in range(len(data)):
    signal = data.loc[wave].dropna().to_numpy(dtype='float32')
    r_peaks = ecg.engzee_segmenter(signal, 300)['rpeaks']
    short_data = ecg.extract_heartbeats(signal, r_peaks, 300)['templates']
    """if (len(short_data)>15):
        print("continue")
        continue"""
    dist_matrix = np.zeros((len(short_data), len(short_data)))
    calculateCenter()
    print(wave)


"""import matplotlib.pyplot as plt
fig, ax = plt.subplots(11,1, figsize=(10,60))
for i in range(len(short_data)):
    ax[i].plot(range(len(short_data[i])), short_data[i])"""


import pickle
with open('centerwaves.pkl', 'wb') as f:
    pickle.dump((centerpoints, centerwaves), f)