import math
import numpy as np
from scipy import stats
from collections import defaultdict

def normalize_data(data):
    """
    Normalize such that the mean of the input is 0 and the sample variance is 1

    :param data: The data set, expressed as a flat list of floats.
    :type data: list

    :return: The normalized data set, as a flat list of floats.
    :rtype: list
    """

    mean = np.mean(data)
    var = 0

    for _ in data:
        data[data.index(_)] = _ - mean

    for _ in data:
        var += math.pow(_, 2)

    var = math.sqrt(var / float(len(data)))

    for _ in data:
        data[data.index(_)] = _ / var

    return data

def ampspaceIndex(x_value,xmin,xmax,phasenums = 4):
    ampinterval = (xmax - xmin) / phasenums
    index = 1
    while xmin < xmax:
        if x_value >= xmin and x_value < xmin + ampinterval:
            return index
        index += 1
        xmin += ampinterval
    return -1

def Flatten(l):
    return [item for sublist in l for item in sublist]

def CombineFeatures(table1, table2):
    '''
    table1 and table2 should have the same length
    '''
    table = []
    n_row = len(table1)
    for i in range(n_row):
        table.append(table1[i]+table2[i])
        
    return table

def RandomNum(ts):
    '''
    baseline feature
    '''
    return [random.random()]

def ThreePointsMedianPreprocess(ts):
    '''
    8-beat sliding window RR interval irregularity detector [21]
    '''
    new_ts = []
    for i in range(len(ts)-2):
        new_ts.append(np.median([ts[i], ts[i+1], ts[i+2]]))
    return new_ts

def ExpAvgPreprocess(ts):
    '''
    8-beat sliding window RR interval irregularity detector [21]
    '''
    alpha = 0.02
    new_ts = [0] * len(ts)
    new_ts[0] = ts[0]
    for i in range(1, len(ts)):
        new_ts[i] = new_ts[i-1] + alpha * (ts[i] - new_ts[i-1])
    
    return new_ts

def Heaviside(x):
    if x < 0:
        return 0.0
    elif x == 0:
        return 0.5
    else:
        return 1.0

def sampen2(data, mm=2, r=0.2, normalize=False):
    """
    Calculates an estimate of sample entropy and the variance of the estimate.

    :param data: The data set (time series) as a list of floats.
    :type data: list

    :param mm: Maximum length of epoch (subseries).
    :type mm: int

    :param r: Tolerance. Typically 0.1 or 0.2.
    :type r: float

    :param normalize: Normalize such that the mean of the input is 0 and
    the sample, variance is 1.
    :type normalize: bool

    :return: List[(Int, Float/None, Float/None)...]

    Where the first (Int) value is the Epoch length.
    The second (Float or None) value is the SampEn.
    The third (Float or None) value is the Standard Deviation.

    The outputs are the sample entropies of the input, for all epoch lengths of
    0 to a specified maximum length, m.

    If there are no matches (the data set is unique) the sample entropy and
    standard deviation will return None.

    :rtype: list
    """

    n = len(data)

    if n == 0:
        raise ValueError("Parameter `data` contains an empty list")

    if mm > n / 2:
        raise ValueError(
            "Maximum epoch length of %d too large for time series of length "
            "%d (mm > n / 2)" % (
                mm,
                n,
            )
        )

    mm += 1

    mm_dbld = 2 * mm

    if mm_dbld > n:
        raise ValueError(
            "Maximum epoch length of %d too large for time series of length "
            "%d ((mm + 1) * 2 > n)" % (
                mm,
                n,
            )
        )

    if normalize is True:
        data = normalize_data(data)

    # initialize the lists
    run = [0] * n
    run1 = run[:]

    r1 = [0] * (n * mm_dbld)
    r2 = r1[:]
    f = r1[:]

    f1 = [0] * (n * mm)
    f2 = f1[:]

    k = [0] * ((mm + 1) * mm)

    a = [0] * mm
    b = a[:]
    p = a[:]
    v1 = a[:]
    v2 = a[:]
    s1 = a[:]
    n1 = a[:]
    n2 = a[:]

    for i in range(n - 1):
        nj = n - i - 1
        y1 = data[i]

        for jj in range(nj):
            j = jj + i + 1

            if data[j] - y1 < r and y1 - data[j] < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]

                for m in range(m1):
                    a[m] += 1
                    if j < n - 1:
                        b[m] += 1
                    f1[i + m * n] += 1
                    f[i + n * m] += 1
                    f[j + n * m] += 1

            else:
                run[jj] = 0

        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]

        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    for i in range(1, mm_dbld):
        for j in range(i - 1):
            r2[i + n * j] = r1[i - j - 1 + n * j]
    for i in range(mm_dbld, n):
        for j in range(mm_dbld):
            r2[i + n * j] = r1[i - j - 1 + n * j]
    for i in range(n):
        for m in range(mm):
            ff = f[i + n * m]
            f2[i + n * m] = ff - f1[i + n * m]
            k[(mm + 1) * m] += ff * (ff - 1)
    m = mm - 1
    while m > 0:
        b[m] = b[m - 1]
        m -= 1
    b[0] = float(n) * (n - 1.0) / 2.0
    for m in range(mm):
        #### added
        if float(b[m]) == 0:
            p[m] = 0.0
            v2[m] = 0.0
        else:
            p[m] = float(a[m]) / float(b[m])
            v2[m] = p[m] * (1.0 - p[m]) / b[m]
    for m in range(mm):
        d2 = m + 1 if m + 1 < mm - 1 else mm - 1
        for d in range(d2):
            for i1 in range(d + 1, n):
                i2 = i1 - d - 1
                nm1 = f1[i1 + n * m]
                nm3 = f1[i2 + n * m]
                nm2 = f2[i1 + n * m]
                nm4 = f2[i2 + n * m]
                # if r1[i1 + n * j] >= m + 1:
                #    nm1 -= 1
                # if r2[i1 + n * j] >= m + 1:
                #    nm4 -= 1
                for j in range(2 * (d + 1)):
                    if r2[i1 + n * j] >= m + 1:
                        nm2 -= 1
                for j in range(2 * d + 1):
                    if r1[i2 + n * j] >= m + 1:
                        nm3 -= 1
                k[d + 1 + (mm + 1) * m] += float(2 * (nm1 + nm2) * (nm3 + nm4))

    n1[0] = float(n * (n - 1) * (n - 2))
    for m in range(mm - 1):
        for j in range(m + 2):
            n1[m + 1] += k[j + (mm + 1) * m]
    for m in range(mm):
        for j in range(m + 1):
            n2[m] += k[j + (mm + 1) * m]

    # calculate standard deviation for the set
    for m in range(mm):
        v1[m] = v2[m]
        ##### added
        if b[m] == 0:
            dv = 0.0
        else:
            dv = (n2[m] - n1[m] * p[m] * p[m]) / (b[m] * b[m])
        if dv > 0:
            v1[m] += dv
        s1[m] = math.sqrt(v1[m])

    # assemble and return the response
    response = []
    for m in range(mm):
        if p[m] == 0:
            # Infimum, the data set is unique, there were no matches.
            response.append((m, None, None))
        else:
            response.append((m, -math.log(p[m]), s1[m]))
    return response

def minimumNCCE(line,phasenums = 4):
    '''
    input: QSR info(R-R interval series) , the number of amplitude part
    method:minimum of the corrected conditional entropy of RR interval sequence
    paper:
    1.Measuring regularity by means of a corrected conditional entropy in sympathetic outflow
    2.Assessment of the dynamics of atrial signals and local atrial period
    series during atrial fibrillation: effects of isoproterenol
    administration(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC529297/)

    :param datas: QSR info from QSRinfo.csv
           phasenums: (xmax - xmin) / phasenums
    :return: [min_NCCE_value,min_L_Index]
    '''
        # normalize
    line = np.array(line, float)
    line = line[1:-1]
    # not enough information
    if line.size == 0:
        return [1, -1]
    line = (line - np.mean(line)) / (np.std(line) + 1e-5)
    # useful param
    N = line.size
    xmin = np.min(line)
    xmax = np.max(line)

    L = 1
    E = defaultdict(float)
    CCE = defaultdict(float)
    NCCE = defaultdict(float)
    while L <= N:
        notsingle_c = defaultdict(int)
        single_c = 0
        for index in range(0, N - L + 1):
            index1 = ampspaceIndex(line[index], xmin, xmax, phasenums)
            index2 = ampspaceIndex(line[index + L - 1], xmin, xmax, phasenums)
            if index1 == index2:
                # not single
                notsingle_c[index1] += 1
            else:
                # single
                single_c += 1

        notsingle_array = np.array(list(notsingle_c.values()), float)
        notsingle_value = np.dot(1 / notsingle_array, np.log(notsingle_array))

        single_value = single_c / N * math.log(N - L + 1)
        E[L] = single_value + notsingle_value
        EcL = single_c / N * E[1]
        CCE[L] = E[L] - E[L - 1] + EcL
        NCCE[L] = CCE[L] / (E[1] + 1e-5)
        L += 1
    CCE_values = np.array(list(CCE.values()))
    minCCE = CCE_values.min()
    minCCEI = CCE_values.argmin()
    NCCE_values = np.array(list(NCCE.values()))
    minNCCE = NCCE_values.min()
    return [minNCCE, minCCEI]

def bin_stat_interval(ts):
    '''
    stat of bin Counter RR interval ts
    
    count, ratio
    '''
    pass

def bin_stat(ts):
    '''
    stat of bin Counter RR ts
    
    count, ratio
    '''
    feature_list.extend(['bin_stat_'+str(i) for i in range(52)])

    if len(ts) > 0:
        interval_1 = [1, 4, 8, 16, 32, 64, 128, 240]
        bins_1 = sorted([240 + i for i in interval_1] + [240 - i for i in interval_1], reverse=True)
        
        cnt_1 = [0.0] * len(bins_1)
        for i in ts:
            for j in range(len(bins_1)):
                if i > bins_1[j]:
                    cnt_1[j] += 1
                    break
        ratio_1 = [i/len(ts) for i in cnt_1]    
        
        interval_2 = [8, 32, 64, 128, 240]
        bins_2 = sorted([240 + i for i in interval_2] + [240 - i for i in interval_2], reverse=True)
        
        cnt_2 = [0.0] * len(bins_2)
        for i in ts:
            for j in range(len(bins_2)):
                if i > bins_2[j]:
                    cnt_2[j] += 1
                    break
        ratio_2 = [i/len(ts) for i in cnt_2]
        
        return cnt_1 + ratio_1 + cnt_2 + ratio_2
    else:
        
        return [0.0] * 52

def drddc(ts):
    '''
    TODO:
    '''
    pass
 
def SampleEn(ts):
    '''    
    sample entropy on QRS interval
    '''
    ts = [float(i) for i in ts]
    mm = 3
    out = []
    feature_list.extend(['SampleEn_'+str(i) for i in range(mm + 1)])

    if len(ts) >= (mm+1)*2:
        res = sampen2(ts, mm=mm, normalize=True)
        for ii in res:
            if ii[1] is None:
                out.append(100)
            else:
                out.append(ii[1])
        return out
    else:
        return [0] * (mm + 1)   
    
def CDF(ts):
    '''
    analysis of cumulative distribution functions [17],
    '''
    n_bins = 60
    hist, _ = np.histogram(ts, range=(100, 400), bins=n_bins)
    cdf = np.cumsum(hist)/len(ts)
    cdf_density = np.sum(cdf) / n_bins
    feature_list.extend(['CDF_cdf_density'])
    return [cdf_density]
    
def CoeffOfVariation(ts):
    '''
    analysis of cumulative distribution functions [17],
    '''
    if len(ts) >= 3:
        tmp_ts = ts[1:-1]
        if np.mean(tmp_ts) == 0:
            coeff_ts = 0.0
        else:
            coeff_ts = np.std(tmp_ts) / np.mean(tmp_ts)
    else:
        coeff_ts = 0.0
    
    if len(ts) >= 4:
        tmp_ts = ts[1:-1]
        tmp_ts = np.diff(tmp_ts)
        if np.mean(tmp_ts) == 0:
            coeff_dts = 0.0
        else:
            coeff_dts = np.std(tmp_ts) / np.mean(tmp_ts)
    else:
        coeff_dts = 0.0
    
    feature_list.extend(['CoeffOfVariation_coeff_ts', 'CoeffOfVariation_coeff_dts'])
    return [coeff_ts, coeff_dts]

def MAD(ts):
    '''
    thresholding on the median absolute deviation (MAD) of RR intervals [18] 
    '''
    ts_median = np.median(ts)
    mad = np.median([np.abs(ii - ts_median) for ii in ts])
    feature_list.extend(['MAD_mad'])
    return [mad]

def QRSBasicStat(ts):
    
    feature_list.extend(['QRSBasicStat_Mean', 
                        'QRSBasicStat_HR', 
                        'QRSBasicStat_Count', 
                        'QRSBasicStat_Range', 
                        'QRSBasicStat_Var', 
                        'QRSBasicStat_Skew', 
                        'QRSBasicStat_Kurtosis', 
                        'QRSBasicStat_Median', 
                        'QRSBasicStat_Min', 
                        'QRSBasicStat_p_5', 
                        'QRSBasicStat_p_25', 
                        'QRSBasicStat_p_75', 
                        'QRSBasicStat_p_95', 
                        'QRSBasicStat_range_95_5', 
                        'QRSBasicStat_range_75_25'])
    
    if len(ts) >= 3:
        
        ts = ts[1:-1]
        
        Mean = np.mean(ts)
        if Mean == 0:
            HR = 0
        else:
            HR = 1 / Mean
        Count = len(ts)
        Range = max(ts) - min(ts)
        Var = np.var(ts)
        Skew = stats.skew(ts)
        Kurtosis = stats.kurtosis(ts)
        Median = np.median(ts)
        Min = min(ts)
        p_5 = np.percentile(ts, 5)
        p_25 = np.percentile(ts, 25)
        p_75 = np.percentile(ts, 75)
        p_95 = np.percentile(ts, 95)
        range_95_5 = p_95 - p_5
        range_75_25 = p_75 - p_25

        return [Mean, HR, Count, Range, Var, Skew, Kurtosis, Median, Min, 
                p_5, p_25, p_75, p_95, 
                range_95_5, range_75_25]
    
    else:
        return [0.0] * 15
    
def QRSBasicStatPointMedian(ts):
    
    feature_list.extend(['QRSBasicStatPointMedian_Mean', 
                         'QRSBasicStatPointMedian_HR', 
                         'QRSBasicStatPointMedian_Count', 
                         'QRSBasicStatPointMedian_Range', 
                         'QRSBasicStatPointMedian_Var', 
                         'QRSBasicStatPointMedian_Skew',
                         'QRSBasicStatPointMedian_Kurtosis',
                         'QRSBasicStatPointMedian_Median',
                         'QRSBasicStatPointMedian_Min',
                         'QRSBasicStatPointMedian_p_25',
                         'QRSBasicStatPointMedian_p_75'])
    
    ts = ThreePointsMedianPreprocess(ts)
    
    Mean = np.mean(ts)
    if Mean == 0:
        HR = 0
    else:
        HR = 1 / Mean

    Count = len(ts)
    if Count != 0:
        Range = max(ts) - min(ts)
        Var = np.var(ts)
        Skew = stats.skew(ts)
        Kurtosis = stats.kurtosis(ts)
        Median = np.median(ts)
        Min = min(ts)
        p_25 = np.percentile(ts, 25)
        p_75 = np.percentile(ts, 75)
    else:
        Range = 0.0
        Var = 0.0
        Skew = 0.0
        Kurtosis = 0.0
        Median = 0.0
        Min = 0.0
        p_25 = 0.0
        p_75 = 0.0
    
    return [Mean, HR, Count, Range, Var, Skew, Kurtosis, Median, Min, p_25, p_75]
      
def QRSBasicStatDeltaRR(ts):
    
    feature_list.extend(['QRSBasicStatDeltaRR_Mean', 
                        'QRSBasicStatDeltaRR_HR', 
                        'QRSBasicStatDeltaRR_Count', 
                        'QRSBasicStatDeltaRR_Range', 
                        'QRSBasicStatDeltaRR_Var', 
                        'QRSBasicStatDeltaRR_Skew', 
                        'QRSBasicStatDeltaRR_Kurtosis', 
                        'QRSBasicStatDeltaRR_Median', 
                        'QRSBasicStatDeltaRR_Min', 
                        'QRSBasicStatDeltaRR_p_25', 
                        'QRSBasicStatDeltaRR_p_75'])
    
    if len(ts) >= 4:
        ts = ts[1:-1]
        ts = np.diff(ts)
        
        Mean = np.mean(ts)
        if Mean == 0:
            HR = 0
        else:
            HR = 1 / Mean
        Count = len(ts)
        Range = max(ts) - min(ts)
        Var = np.var(ts)
        Skew = stats.skew(ts)
        Kurtosis = stats.kurtosis(ts)
        Median = np.median(ts)
        Min = min(ts)
        p_25 = np.percentile(ts, 25)
        p_75 = np.percentile(ts, 75)
        return [Mean, HR, Count, Range, Var, Skew, Kurtosis, Median, Min, p_25, p_75]
    
    else:
        return [0.0] * 11
    
def QRSYuxi(ts):
    '''
    pars: 
        tol = 0.05
            define if two QRS intervals are matched
    '''
    tol = 0.05
    feature_list.extend(['QRSYuxi'])

    
    if len(ts) >= 3:
        ts = ts[1:-1]
    
        avg_RR = np.median(ts)
        matched = [False] * len(ts)
        
        for i in range(len(ts)):
            seg_1 = ts[i]
            if abs(seg_1 - avg_RR) / avg_RR <= tol:
                matched[i] = True
            elif abs(seg_1 - 2 * avg_RR) / (2 * avg_RR) <= tol:
                matched[i] = True
                
        for i in range(len(ts)):
            if matched[i] is False:
                if i == 0:
                    seg_2_forward = ts[i]
                else:
                    seg_2_forward = ts[i-1] + ts[i]
                if i == len(ts)-1:
                    seg_2_backward = ts[i]
                else:
                    seg_2_backward = ts[i] + ts[i+1]
                    
                if abs(seg_2_forward - 2 * avg_RR) / (2 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_forward - 3 * avg_RR) / (3 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_backward - 2 * avg_RR) / (2 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_backward - 3 * avg_RR) / (3 * avg_RR) <= tol:
                    matched[i] = True
    
        return [sum(matched) / len(matched)]

    else:
        return [0.0] * 1
    
def Variability(ts):
    '''
    Variability(Time Domain) & Poincare plot
    https://zh.wikipedia.org/wiki/%E5%BF%83%E7%8E%87%E8%AE%8A%E7%95%B0%E5%88%86%E6%9E%90
    compute SDNN, NN50 count, pNN50
    [14] Atrial fibrillation detection by heart rate variability in Poincare plot
    Stepping: the mean stepping increment of the inter-beat intervals
    Dispersion: how spread the points in Poincaré plot are distributed around the diagonal line
    '''
    feature_list.extend(['Variability_SDNN', 
                        'Variability_NN50', 
                        'Variability_pNN50', 
                        'Variability_Stepping', 
                        'Variability_Dispersion'])

    if len(ts) >= 3:
        ts = ts[1:-1]
        SDNN = np.std(ts)
        freq = 300
        timelen = freq * 0.05
        if len(ts) < 3:
            NN50 = 0
            pNN50 = 0
            Stepping = 0
            Dispersion = 0
        else:
            NN = [abs(ts[x + 1] - ts[x]) for x in range(len(ts) - 1)]
            NN50 = sum([x > timelen for x in NN])
            pNN50 = float(NN50) / len(ts)
            Stepping = (sum([(NN[x] ** 2 + NN[x + 1] ** 2) ** 0.5 for x in range(len(NN) - 1)]) / (len(NN) - 1)) / (sum(ts) / len(ts))
            Dispersion = (sum([x ** 2 for x in NN]) / (2 * len(NN)) - sum(NN) ** 2 / (2 * (len(NN)) ** 2)) ** 0.5 / ((-ts[0] - 2 * ts[-1] + 2 * sum(ts)) / (2 * len(NN)))
        return [SDNN, NN50, pNN50, Stepping, Dispersion]
    
    else:
        return [0.0] * 5

feature_list = []
row = []

def GetQRSFeatures(ts):
    row.extend(SampleEn(ts))
    row.extend(CDF(ts))
    row.extend(CoeffOfVariation(ts))
    row.extend(MAD(ts))
    row.extend(QRSBasicStat(ts))
    row.extend(QRSBasicStatPointMedian(ts))
    row.extend(QRSBasicStatDeltaRR(ts))
    row.extend(QRSYuxi(ts))
    row.extend(Variability(ts))
    row.extend(minimumNCCE(ts))
    row.extend(bin_stat(ts))

    

    return row, feature_list
