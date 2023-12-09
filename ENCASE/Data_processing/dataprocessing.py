import numpy
import pickle
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def processPickle(inData, trainData, testData):
    with open(inData, 'rb') as file:
        x_data, y_data = pickle.load(file)

    zipped = list(zip(x_data, y_data))
    random.shuffle(zipped)
    train_val_len = int(len(zipped)*0.8)
    train_val = zipped[:train_val_len]
    test = zipped[train_val_len:]

    with open(trainData, 'wb') as file:
        pickle.dump(zip(*train_val), file)
    with open(testData, 'wb') as file:
        pickle.dump(zip(*test), file)

def processCSV(inDataX, inDataY, prefix, train_size):
    df1 = pd.read_csv(inDataX, index_col='id')
    df2 = pd.read_csv(inDataY, index_col='id')

    indices = df1.index.tolist()
    random.shuffle(indices)

    df1 = df1.reindex(indices)
    df2 = df2.reindex(indices)

    df1_train, df1_test = train_test_split(df1, train_size=train_size, shuffle=False)
    df2_train, df2_test = train_test_split(df2, train_size=train_size, shuffle=False)

    df1_train.to_csv(prefix + "_train_x.csv")
    df1_test.to_csv(prefix + "_valid_x.csv")
    df2_train.to_csv(prefix + "_train_y.csv")
    df2_test.to_csv(prefix + "_valid_y.csv")