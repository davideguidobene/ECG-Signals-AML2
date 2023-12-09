import math
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb 
from scipy import stats
import matplotlib.pyplot as plt 
import biosppy.signals.ecg as ecg
from scipy.signal import periodogram

from Data_processing.data_expander import expandData
from Data_processing.dataprocessing import processCSV

from DNN1 import DNN1
from DNN1.DNN1 import getDeepFeatures
from Feature_extraction.features_short import getShortFeatures
from Feature_extraction.features_long import getLongFeatures

### GLOBAL VARS ###########################################
#x_data = "Data/X_train.csv"
#y_data = "Data/y_train.csv"

x_data = "C:\\Users\\giova\\Desktop\\aml_prj\\task2\\AML_project2\\X_train.csv"
y_data = "C:\\Users\\giova\\Desktop\\aml_prj\\task2\\AML_project2\\y_train.csv"



splitted_prefix = "Data/splitted"
splitted_train_x = "Data/splitted_train_x.csv"
splitted_train_y = "Data/splitted_train_y.csv"
splitted_valid_x = "Data/splitted_valid_x.csv"
splitted_valid_y = "Data/splitted_valid_y.csv"

trainset_size = 0.8

evenet_train = "Data/evenet_train.pkl"
evenet_valid = "Data/evenet_valid.pkl"

#last_checkpoint="~/AML_project2/lightning_logs/version_36294090/checkpoints/epoch=29-step=29280.ckpt"
last_checkpoint = "C:\\Users\\giova\\Desktop\\aml_prj\\task2\\AML_project2\\lightning_logs\\version_36294090\\checkpoints\\epoch=29-step=29280.ckpt"


### TRAINING ###############################################

def setup():
    """
    ###divide train from valid data
    processCSV(x_data, y_data, splitted_prefix, trainset_size)

    ###data expanding and processing for DNN1
    expandData(splitted_train_x, splitted_train_y, evenet_train)
    expandData(splitted_valid_x, splitted_valid_y, evenet_valid)

    ###train DNN1
    DNN1.train(evenet_train, evenet_valid, last_checkpoint, True)
    """
    #DNN1.train(evenet_train, evenet_valid, last_checkpoint, False)

    ###train xgb
    trainingFeatures = []
    df1 = pd.read_csv(x_data, index_col='id')
    df2 = pd.read_csv(y_data, index_col='id')
    for i in range(len(df1)):
        trainingRound = []
        trainingRound += getShortFeatures(df1.iloc[i].dropna())
        trainingRound += getLongFeatures(df1.iloc[i].dropna())
        trainingRound += getDeepFeatures(df1.iloc[i].dropna(), last_checkpoint)
        trainingFeatures.append(trainingRound)

    dtrain = xgb.DMatrix(trainingFeatures, label=df2.iloc[:,0])
    params = {
        'objective': 'multi:softmax',  # for multi-class classification
        'num_class': 4,               # number of classes
        'max_depth': 3,               # maximum depth of individual trees
        'learning_rate': 0.1,        # learning rate
        'n_estimators': 100,          # number of boosting rounds
    }
    num_boost_round = 100  # Number of boosting rounds
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

def classify(data):

    ###cross validation on evenet_valid


    ###prediction
    getDeepFeatures(il_dato, last_checkpoint)


        

    dvalid = xgb.DMatrix(X_valid)
    y_pred = model.predict(dvalid)

setup()