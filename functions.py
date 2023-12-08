import datetime
from statistics import mean, median, stdev
import numpy as np
import neurokit2 as nk
import pandas as pd
from biosppy.signals import ecg


def calc_R_period(signal, r_peaks, measurements):
    r_onset = measurements['ECG_R_Onsets']
    r_offset = measurements['ECG_R_Offsets']
    periods = []
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope = []
    for i in range(len(r_peaks)):
        if not np.isnan(r_onset[i]) and not np.isnan(r_offset[i]):
            periods.append(r_offset[i] - r_onset[i])
            amplitudes_onset.append(signal[r_onset[i]])
            amplitudes_offset.append(signal[r_offset[i]])
            amplitudes_diff.append(signal[r_offset[i]] - signal[r_onset[i]])
            amplitudes_slope.append((signal[r_offset[i]] - signal[r_onset[i]])/(r_offset[i] - r_onset[i]))

    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff), np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_R_amplitude(signal, r_peaks, measurements):
    amplitudes = []
    for peak in r_peaks:
        amplitudes.append(signal[peak])
    amplitudes = np.array(amplitudes)
    if len(amplitudes):
        return np.mean(amplitudes),np.median(amplitudes), np.std(amplitudes), np.max(amplitudes), np.min(amplitudes), np.max(amplitudes) - np.min(amplitudes), np.mean(amplitudes) - np.median(amplitudes)
    else:
        return 7*[0.0]


def calc_Q_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_Q_Peaks']
    amplitudes = []
    positions = []
    index = 0
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
            positions.append(abs(r_peaks[index] - peak) )
        index += 1
    amplitudes = np.array(amplitudes)
    positions = np.array(positions)
    if len(amplitudes):
        return np.mean(amplitudes),np.median(amplitudes), np.std(amplitudes), np.max(amplitudes), np.min(amplitudes),  np.max(amplitudes) - np.min(amplitudes), np.mean(amplitudes) - np.median(amplitudes), np.mean(positions),np.median(positions), np.std(positions), np.max(positions), np.min(positions), np.max(positions) - np.min(positions), np.mean(positions) - np.median(positions)
    else:
        return 14 * [0.0]


def calc_S_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_S_Peaks']
    amplitudes = []
    positions = []
    index = 0
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
            positions.append(abs(r_peaks[index] - peak) )
        index += 1
    amplitudes = np.array(amplitudes)
    positions = np.array(positions)
    if len(amplitudes):
        return np.mean(amplitudes),np.median(amplitudes), np.std(amplitudes), np.max(amplitudes), np.min(amplitudes), np.max(amplitudes) - np.min(amplitudes), np.mean(amplitudes) - np.median(amplitudes), np.mean(positions),np.median(positions), np.std(positions),  np.max(positions), np.min(positions), np.max(positions) - np.min(positions), np.mean(positions) - np.median(positions)
    else:
        return 14 * [0.0]


def calc_T_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_T_Peaks']
    amplitudes = []
    positions = []
    index = 0
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
            positions.append(abs(r_peaks[index] - peak) )
        index += 1
    amplitudes = np.array(amplitudes)
    if len(amplitudes):
        return np.mean(amplitudes),np.median(amplitudes), np.std(amplitudes), np.max(amplitudes), np.min(amplitudes), np.max(amplitudes) - np.min(amplitudes), np.mean(amplitudes) - np.median(amplitudes), np.mean(positions),np.median(positions), np.std(positions),  np.max(positions), np.min(positions), np.max(positions) - np.min(positions), np.mean(positions) - np.median(positions)
    else:
        return 14 * [0.0]


def calc_P_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_P_Peaks']
    amplitudes = []
    positions = []
    index = 0
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
            positions.append(abs(r_peaks[index] - peak) )
        index += 1
    amplitudes = np.array(amplitudes)
    if len(amplitudes):
        return np.mean(amplitudes),np.median(amplitudes), np.std(amplitudes), np.max(amplitudes), np.min(amplitudes), np.min(amplitudes), np.max(amplitudes) - np.min(amplitudes), np.mean(amplitudes) - np.median(amplitudes),np.mean(positions),np.median(positions), np.std(positions),  np.max(positions), np.min(positions), np.max(positions) - np.min(positions), np.mean(positions) - np.median(positions)
    else:
        return 14 * [0.0]


def calc_T_period(signal, r_peaks, measurements):
    t_onset = measurements['ECG_T_Onsets']
    t_offset = measurements['ECG_T_Offsets']
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope = []
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(t_onset[i]) and not np.isnan(t_offset[i]):
            periods.append(t_offset[i] - t_onset[i])
            amplitudes_onset.append(signal[t_onset[i]])
            amplitudes_offset.append(signal[t_offset[i]])
            amplitudes_diff.append(signal[t_offset[i]] - signal[t_onset[i]])
            amplitudes_slope.append((signal[t_offset[i]] - signal[t_onset[i]])/(t_offset[i] - t_onset[i]))
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff),  np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_P_period(signal, r_peaks, measurements):
    p_onset = measurements['ECG_P_Onsets']
    p_offset = measurements['ECG_P_Offsets']
    periods = []
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope = []
    for i in range(len(r_peaks)):
        if not np.isnan(p_onset[i]) and not np.isnan(p_offset[i]):
            periods.append(p_offset[i] - p_onset[i])
            amplitudes_onset.append(signal[p_onset[i]])
            amplitudes_offset.append(signal[p_offset[i]])
            amplitudes_diff.append(signal[p_offset[i]] - signal[p_onset[i]])
            amplitudes_slope.append((signal[p_offset[i]] - signal[p_onset[i]])/(p_offset[i] - p_onset[i]))
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff), np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_Q_period(signal, r_peaks, measurements):
    r_onset = measurements['ECG_R_Onsets']
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    periods = []
    amplitudes_slope = []
    for i in range(len(r_peaks)):
        if not np.isnan(r_onset[i]):
            periods.append(r_peaks[i] - r_onset[i])
            amplitudes_onset.append(signal[r_onset[i]])
            amplitudes_offset.append(signal[r_peaks[i]])
            amplitudes_diff.append(signal[r_peaks[i]] - signal[r_onset[i]])
            amplitudes_slope.append((signal[r_peaks[i]] - signal[r_onset[i]])/(r_peaks[i] - r_onset[i]))
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff),np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_S_period(signal, r_peaks, measurements):
    r_offset = measurements['ECG_R_Offsets']
    periods = []
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope = []
    for i in range(len(r_peaks)):
        if not np.isnan(r_offset[i]):
            periods.append(r_offset[i] - r_peaks[i])
            amplitudes_onset.append(signal[r_peaks[i]])
            amplitudes_offset.append(signal[r_offset[i]])
            amplitudes_diff.append(signal[r_offset[i]] - signal[r_peaks[i]])
            amplitudes_slope.append((signal[r_offset[i]] - signal[r_peaks[i]])/r_offset[i] - r_peaks[i])
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff), np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_PR_interval(signal, r_peaks, measurements):
    p_onset = measurements['ECG_P_Onsets']
    r_onset = measurements['ECG_R_Onsets']
    periods = []
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope = []
    for i in range(len(r_peaks)):
        if not np.isnan(p_onset[i]) and not np.isnan(r_onset[i]):
            periods.append(r_onset[i] - p_onset[i])
            amplitudes_onset.append(signal[p_onset[i]])
            amplitudes_offset.append(signal[r_onset[i]])
            amplitudes_diff.append(signal[r_onset[i]] - signal[p_onset[i]])
            amplitudes_slope.append((signal[r_onset[i]] - signal[p_onset[i]])/(r_onset[i] - p_onset[i]))
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff), np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_QT_interval(signal, r_peaks, measurements):
    t_offset = measurements['ECG_T_Offsets']
    r_onset = measurements['ECG_R_Onsets']
    periods = []
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope = []
    for i in range(len(r_peaks)):
        if not np.isnan(t_offset[i]) and not np.isnan(r_onset[i]):
            periods.append(t_offset[i] - r_onset[i])
            amplitudes_onset.append(signal[r_onset[i]])
            amplitudes_offset.append(signal[t_offset[i]])
            amplitudes_diff.append(signal[t_offset[i]] - signal[r_onset[i]])
            amplitudes_slope.append((signal[t_offset[i]] - signal[r_onset[i]])/(t_offset[i] - r_onset[i]))
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff), np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_PR_segment(signal, r_peaks, measurements):
    p_offset = measurements['ECG_P_Offsets']
    r_onset = measurements['ECG_R_Onsets']
    periods = []
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope = []
    for i in range(len(r_peaks)):
        if not np.isnan(p_offset[i]) and not np.isnan(r_onset[i]):
            periods.append(r_onset[i] - p_offset[i])
            amplitudes_onset.append(signal[r_onset[i]])
            amplitudes_offset.append(signal[p_offset[i]])
            amplitudes_diff.append(signal[r_onset[i]] - signal[p_offset[i]])
            amplitudes_slope.append((signal[r_onset[i]] - signal[p_offset[i]])/(r_onset[i] - p_offset[i]))
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff), np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0]


def calc_ST_segment(signal, r_peaks, measurements):
    t_onset = measurements['ECG_T_Onsets']
    r_offset = measurements['ECG_R_Offsets']
    periods = []
    amplitudes_onset = []
    amplitudes_offset = []
    amplitudes_diff = []
    amplitudes_slope=[]
    for i in range(len(r_peaks)):
        if not np.isnan(r_offset[i]) and not np.isnan(t_onset[i]):
            periods.append(t_onset[i] - r_offset[i])
            amplitudes_onset.append(signal[t_onset[i]])
            amplitudes_offset.append(signal[r_offset[i]])
            amplitudes_diff.append(signal[t_onset[i]] - signal[r_offset[i]])
            amplitudes_slope.append((signal[t_onset[i]] - signal[r_offset[i]])/(t_onset[i] - r_offset[i]))
    periods = np.array(periods)
    if len(periods):
        return np.mean(periods),np.median(periods), np.std(periods), np.max(periods), np.min(periods), np.max(periods) - np.min(periods), np.mean(periods) - np.median(periods),np.mean(amplitudes_onset),np.median(amplitudes_onset), np.std(amplitudes_onset), np.max(amplitudes_onset), np.min(amplitudes_onset), np.max(amplitudes_onset) - np.min(amplitudes_onset), np.mean(amplitudes_onset) - np.median(amplitudes_onset), np.mean(amplitudes_offset),np.median(amplitudes_offset), np.std(amplitudes_offset), np.max(amplitudes_offset), np.min(amplitudes_offset), np.max(amplitudes_offset) - np.min(amplitudes_offset), np.mean(amplitudes_offset) - np.median(amplitudes_offset), np.mean(amplitudes_diff),np.median(amplitudes_diff), np.std(amplitudes_diff), np.max(amplitudes_diff), np.min(amplitudes_diff), np.max(amplitudes_diff) - np.min(amplitudes_diff), np.mean(amplitudes_diff) - np.median(amplitudes_diff),np.mean(amplitudes_slope),np.median(amplitudes_slope), np.std(amplitudes_slope), np.max(amplitudes_slope), np.min(amplitudes_slope), np.max(amplitudes_slope) - np.min(amplitudes_slope), np.mean(amplitudes_slope) - np.median(amplitudes_slope)
    else:
        return 35 * [0.0] 

def calc_RR_intervals(signal, r_peaks,measurements):
    rr_intervals = np.diff(r_peaks)
    delta_y = np.diff(signal[r_peaks])
    slope = delta_y/rr_intervals

    if len(rr_intervals):
        return np.mean(rr_intervals),np.median(rr_intervals), np.std(rr_intervals), np.max(rr_intervals), np.min(rr_intervals), np.max(rr_intervals) - np.min(rr_intervals), np.mean(rr_intervals) - np.median(rr_intervals), np.log(np.mean(rr_intervals)), np.log(np.median(rr_intervals)), np.mean(slope), np.median(slope), np.std(slope), np.max(slope), np.min(slope), np.max(slope) - np.min(slope), np.mean(slope) - np.median(slope), np.mean(delta_y), np.median(delta_y), np.std(delta_y), np.max(delta_y), np.min(delta_y), np.max(delta_y) - np.min(delta_y), np.mean(delta_y) - np.median(delta_y)
    else:
        return 23 * [0.0]

def get_nk_features(signal, r_peaks, measurements):
    features = np.concatenate((
        calc_R_period(signal, r_peaks, measurements),
        calc_R_amplitude(signal, r_peaks, measurements),
        calc_Q_amplitude(signal, r_peaks, measurements),
        calc_S_amplitude(signal, r_peaks, measurements),
        calc_T_amplitude(signal, r_peaks, measurements),
        calc_P_amplitude(signal, r_peaks, measurements),
        calc_T_period(signal, r_peaks, measurements),
        calc_P_period(signal, r_peaks, measurements),
        calc_Q_period(signal, r_peaks, measurements),
        calc_S_period(signal, r_peaks, measurements),
        calc_PR_interval(signal, r_peaks, measurements),
        calc_QT_interval(signal, r_peaks, measurements),
        calc_PR_segment(signal, r_peaks, measurements),
        calc_RR_intervals(signal, r_peaks, measurements),
        calc_ST_segment(signal, r_peaks, measurements)),axis=None)
        
    return features

def add_basic_info(row, arr, name):
    if len(arr) > 0:
        row[f'std_{name}'] = arr.std()
        row[f'mean_{name}'] = arr.mean()
        row[f'median_{name}'] = np.median(arr)
        row[f'max_{name}'] = arr.max()
        row[f'min_{name}'] = arr.min()
        row[f'range_{name}'] = arr.max() - arr.min()
    return row
