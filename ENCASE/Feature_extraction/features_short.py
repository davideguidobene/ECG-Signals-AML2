import numpy as np
from scipy import stats
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt 

def ShortZeroCrossing(ts):
    cnt = 0
    for i in range(len(ts)-1):
        if ts[i] * ts[i+1] < 0:
            cnt += 1
        if ts[i] == 0 and ts[i-1] * ts[i+1] < 0:
            cnt += 1
    return cnt

def LongThresCrossing(ts, thres):
    cnt = 0
    pair_flag = 1
    pre_loc = 0
    width = []
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
            if pair_flag == 1:
                width.append(i-pre_loc)
                pair_flag = 0
            else:
                pair_flag = 1
                pre_loc = i
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    
    if len(width) > 1:
        return [cnt, np.mean(width)]
    else:
        return [cnt, 0.0]

def ShortFeatures(ts,prev_ts):
    #stats
    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)
    ZeroCrossing = ShortZeroCrossing(ts)
    
    # medical features
    T_start = round(0.15 * len(ts))
    T_end = round(0.55 * len(ts))
    P_start = round(0.65 * len(ts))
    P_end = round(0.95 * len(ts))
    
    T_wave = ts[T_start:T_end]
    P_wave = ts[P_start:P_end]
    
    T_peak = max(T_wave)
    P_peak = max(P_wave)
    Q_peak = min(prev_ts[-40:])
    R_peak = ts[0]
    S_peak = min(ts[:40])
    
    T_loc = np.argmax(T_wave)
    P_loc = np.argmax(P_wave)

    #Q_loc = -np.argmin(prev_ts[-40:])
    Q_loc = - 40 + np.argmin(prev_ts[-40:])

    R_loc = 0
    S_loc = np.argmin(ts[:40])

    """if Q_loc <= T_loc:
        Q_loc = len(ts) - 20
        print("exception")"""

    #PR_interval = P_loc - 0
    PR_interval = P_end - P_start

    QRS_duration = S_loc - Q_loc

    #QT_interval = T_loc - Q_loc
    QT_interval = T_end - Q_loc


    QT_corrected = QT_interval / len(ts)

    #TQ_interval = ts[T_loc:Q_loc]
    new_Q_loc = int(0.95*len(ts)) + np.argmin(ts[-1* int(0.05*len(ts)):])
    TQ_interval = ts[T_end:new_Q_loc]

    thres = np.mean(TQ_interval) + (T_peak - np.mean(TQ_interval))/50
    NF, Fwidth = LongThresCrossing(TQ_interval, thres)
    
    RQ_amp = R_peak - Q_peak
    RS_amp = R_peak - S_peak
    ST_amp = T_peak - S_peak
    PQ_amp = P_peak - Q_peak
    QS_amp = Q_peak - S_peak
    RP_amp = R_peak - P_peak
    RT_amp = R_peak - T_peak
    
    #ST_interval = T_loc - S_loc
    ST_interval = T_start - S_loc
    # alternative interval: T_start + T_loc - S_loc

    RS_interval = S_loc - R_loc

    return [Range,Var,Skew,Kurtosis,Median,ZeroCrossing,PR_interval,QRS_duration,QT_interval,QT_corrected,NF,Fwidth,
    RQ_amp,RS_amp,ST_amp,PQ_amp,QS_amp,RP_amp,RT_amp,ST_interval,RS_interval]

def getShortFeatures(signal):
    seg = ecg.engzee_segmenter(signal,300)
    r_peaks = seg["rpeaks"]
    ###short_data = ecg.extract_heartbeats(signal, r_peaks, 300)['templates']
    short_data = []
    for i in range(len(r_peaks)-1):
        short_data.append(signal[r_peaks[i]:r_peaks[i+1]])

    num_features = 21
    l = len(short_data[1:])

    results = np.zeros((l,num_features))

    for i in range(l):
        #print(ShortFeatures(short_data[i+1],short_data[i]))
        #print("ok")
        #print(ShortFeatures(short_data[i+1],short_data[i])[10])
        #return
        results[i,:] = ShortFeatures(short_data[i+1],short_data[i])

    print(results)

    final_features_mean = np.mean(results,axis=0)

    features_list = original_array = ["Range", "Var", "Skew", "Kurtosis", "Median", "ZeroCrossing", "PR_interval", "QRS_duration",
     "QT_interval", "QT_corrected", "TQ_interval", "RQ_amp", "RS_amp", "ST_amp", "PQ_amp", "QS_amp", "RP_amp", "RT_amp", "ST_interval", "RS_interval"]

    return final_features_mean,features_list


