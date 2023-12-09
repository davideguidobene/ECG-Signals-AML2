import pandas as pd
from features_long import GetLongFeatures
from features_short import GetShortFeatures
from features_qrs import GetQRSFeatures
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")

data = pd.read_csv('X_train_short.csv', index_col='id')

"""for i in range(10):
    signal = data.loc[i].dropna().to_numpy(dtype='float32')
    GetShortFeatures(signal)"""

"""signal = data.loc[9].dropna().to_numpy(dtype='float32')
print(GetLongFeatures(signal))
print(GetQRSFeatures(signal))
print(GetShortFeatures(signal))"""