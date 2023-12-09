import math
import numpy as np
from scipy import stats
from scipy.signal import periodogram

def WaveletStat(ts):
    '''
    Statistic features for DWT
    '''
    DWTfeat = []
    feature_list.extend(['WaveletStat_'+str(i) for i in range(48)])
    if len(ts) >= 1664:
        db7 = pywt.Wavelet('db7')      
        cAcD = pywt.wavedec(ts, db7, level = 7)
        for i in range(8):
            DWTfeat = DWTfeat + [max(cAcD[i]), min(cAcD[i]), np.mean(cAcD[i]),
                                    np.median(cAcD[i]), np.std(cAcD[i])]
            energy = 0
            for j in range(len(cAcD[i])):
                energy = energy + cAcD[i][j] ** 2
            DWTfeat.append(energy/len(ts))
        return DWTfeat
    else:
        return [0.0]*48
    
def LongBasicStat(ts):

    '''
    TODO: 
    
    1. why too much features will decrease F1
    2. how about add them and feature filter before xgb
    
    '''
    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)
#    p_001 = np.percentile(ts, 0.01)
#    p_002 = np.percentile(ts, 0.02)
#    p_005 = np.percentile(ts, 0.05)
#    p_01 = np.percentile(ts, 0.1)
#    p_02 = np.percentile(ts, 0.2)
#    p_05 = np.percentile(ts, 0.5)
    p_1 = np.percentile(ts, 1)
#    p_2 = np.percentile(ts, 2)
    p_5 = np.percentile(ts, 5)
    p_10 = np.percentile(ts, 10)
    p_25 = np.percentile(ts, 25)
    p_75 = np.percentile(ts, 75)
    p_90 = np.percentile(ts, 90)
    p_95 = np.percentile(ts, 95)
#    p_98 = np.percentile(ts, 98)
    p_99 = np.percentile(ts, 99)
#    p_995 = np.percentile(ts, 99.5)
#    p_998 = np.percentile(ts, 99.8)
#    p_999 = np.percentile(ts, 99.9)
#    p_9995 = np.percentile(ts, 99.95)
#    p_9998 = np.percentile(ts, 99.98)
#    p_9999 = np.percentile(ts, 99.99)

    range_99_1 = p_99 - p_1
    range_95_5 = p_95 - p_5
    range_90_10 = p_90 - p_10
    range_75_25 = p_75 - p_25
    
#    return [Range, Var, Skew, Kurtosis, Median]

#    return [Range, Var, Skew, Kurtosis, Median, 
#            p_1, p_5, p_95, p_99]
    
    feature_list.extend(['LongBasicStat_Range', 
                         'LongBasicStat_Var', 
                        'LongBasicStat_Skew', 
                        'LongBasicStat_Kurtosis', 
                        'LongBasicStat_Median', 
                        'LongBasicStat_p_1', 
                        'LongBasicStat_p_5', 
                        'LongBasicStat_p_95', 
                        'LongBasicStat_p_99', 
                        'LongBasicStat_p_10', 
                        'LongBasicStat_p_25', 
                        'LongBasicStat_p_75', 
                        'LongBasicStat_p_90', 
                        'LongBasicStat_range_99_1', 
                        'LongBasicStat_range_95_5', 
                        'LongBasicStat_range_90_10', 
                        'LongBasicStat_range_75_25'])
    return [Range, Var, Skew, Kurtosis, Median, 
            p_1, p_5, p_95, p_99, 
            p_10, p_25, p_75, p_90, 
            range_99_1, range_95_5, range_90_10, range_75_25]

def LongZeroCrossing(ts, thres):
    cnt = 0
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    feature_list.extend(['LongZeroCrossing_cnt'])
    return [cnt]
    
def LongFFTBandPower(ts):
    '''
    return list of power of each freq band
    
    TODO: different band cut method
    '''
    fs = 300
    nfft = len(ts)
    partition = [0, 1.5, 4, 8, 20, 100, fs/2]
    f, psd = periodogram(ts, fs)
    partition = [int(x * nfft / fs) for x in partition]
    p = [sum(psd[partition[x] : partition[x + 1]]) for x in range(len(partition)-1)]
    
    feature_list.extend(['LongFFTBandPower_'+str(i) for i in range(len(p))])

    return p

def LongFFTPower(ts):
    '''
    return power
    
    no effect
    '''
    psd = periodogram(ts, fs=300.0, nfft=4500)
    power = np.sum(psd[1])
    feature_list.extend(['LongFFTPower_power'])
    return [power]

def LongFFTBandPowerShannonEntropy(ts):
    '''
    return entropy of power of each freq band
    refer to scipy.signal.periodogram
    
    TODO: different band cut method
    '''
    fs = 300
    nfft = len(ts)
    partition = [0, 1.5, 4, 8, 20, 100, fs/2]
    f, psd = periodogram(ts, fs)
    partition = [int(x * nfft / fs) for x in partition]
    p = [sum(psd[partition[x] : partition[x + 1]]) for x in range(len(partition)-1)]
    prob = [x / sum(p) for x in p]
    entropy = sum([- x * math.log(x) for x in prob])
    feature_list.extend(['LongFFTBandPowerShannonEntropy_entropy'])
    return [entropy]

def LongSNR(ts):

    '''
    TODO
    '''
    psd = periodogram(ts, fs=300.0)

    signal_power = 0
    noise_power = 0
    for i in range(len(psd[0])):
        if psd[0][i] < 5:
            signal_power += psd[1][i]
        else:
            noise_power += psd[1][i]
          
    feature_list.extend(['LongSNR_snr'])
      
    return [signal_power / noise_power]

feature_list = []
row = []

def getLongFeatures(ts):
    row.extend(LongBasicStat(ts))
    row.extend(LongZeroCrossing(ts,0))
    row.extend(LongFFTBandPower(ts))
    row.extend(LongFFTPower(ts))
    row.extend(LongFFTBandPowerShannonEntropy(ts))
    row.extend(LongSNR(ts))
    return row, feature_list


