import numpy as np
from scipy import stats
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt 

def CenterwaveZeroCrossing(ts):
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

def CenterwaveFeatures(ts):
    #stats
    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)
    ZeroCrossing = CenterwaveZeroCrossing(ts)
    
    # medical features
    R_loc = np.argmax(ts)

    R_percentage = R_loc / len(ts)


    T_start = round((0.1 + R_percentage) * len(ts))
    T_end = round(min((0.5 + R_percentage),1)*len(ts))
    P_start = round(0.05 * len(ts))
    P_end = round((max(R_percentage - 0.05,0.06) )* len(ts))

    T_wave = ts[T_start:T_end]
    P_wave = ts[P_start:P_end]

    T_peak = max(T_wave)
    P_peak = max(P_wave)

    T_loc = np.argmax(T_wave)
    P_loc = np.argmax(P_wave)

    Q_peak = min(ts[R_loc-40:R_loc])
    R_peak = ts[R_loc]
    S_peak = min(ts[R_loc:R_loc+40])

    Q_loc = R_loc -40 + np.argmin(ts[R_loc-40:R_loc])
    S_loc = R_loc + np.argmin(ts[R_loc:R_loc+40])

    PR_interval = Q_loc - P_start

    QRS_duration = S_loc - Q_loc

    QT_interval = T_end - Q_loc

    QT_corrected = QT_interval / len(ts)

    
    
    
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

    return [Range,Var,Skew,Kurtosis,Median,ZeroCrossing,PR_interval,QRS_duration,QT_interval,QT_corrected,
    RQ_amp,RS_amp,ST_amp,PQ_amp,QS_amp,RP_amp,RT_amp,ST_interval,RS_interval]

def GetCenterwaveFeatures(centerwave):
    centerpoints, Centerwaves = centerwave

    final_features_mean = CenterwaveFeatures(centerwave)

    features_list = ["Range", "Var", "Skew", "Kurtosis", "Median", "ZeroCrossing", "PR_interval", "QRS_duration",
     "QT_interval", "QT_corrected",  "RQ_amp", "RS_amp", "ST_amp", "PQ_amp", "QS_amp", "RP_amp", "RT_amp", "ST_interval", "RS_interval"]

    return final_features_mean,features_list
