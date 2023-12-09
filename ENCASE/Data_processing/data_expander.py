import numpy as np
import pandas as pd
import pickle


def create_new_data(signal,ws,overlap):
    ret = []
    if overlap > ws:
        raise(ValueError)
    j = 0
    stride = int(ws- overlap)
    for i in range(int(np.floor((len(signal)-ws)/stride)+1)):
        ret.append(signal[j:j+ws])
        j = j+stride
    return ret

def expandData(pathX, pathY, pathOutput): 
    ws = 1024
    overlap = 100

    data = pd.read_csv(pathX, index_col='id')
    labels = pd.read_csv(pathY, index_col='id')
    labels_array = labels['y'].to_numpy()
        
    class_0 = np.argwhere(labels_array==0).flatten()
    class_1 = np.argwhere(labels_array==1).flatten()
    class_2 = np.argwhere(labels_array==2).flatten()
    class_3 = np.argwhere(labels_array==3).flatten()
    n_0, n_1, n_2, n_3 = len(class_0),len(class_1),len(class_2),len(class_3)
    #print(len(class_0),len(class_1),len(class_2),len(class_3))


    classes = []
    new_data = []
    stride_0 = 800
    stride_1 = stride_0 // (n_0 / n_1)
    stride_2 = stride_0 // (n_0 / n_2)
    stride_3 = 0.8* stride_0 // (n_0 / n_3)
    strides = [stride_0,stride_1,stride_2,stride_3]
 
    for i in range(len(data)):
        ##classe = [1,0,0,0] if i in class_0 else [0,1,0,0] if i in class_1 else [0,0,1,0] if i in class_2 else [0,0,0,1]
        ##new_samples = create_new_data(data.loc[i].dropna().to_numpy(),ws, ws- strides[classe.index(1)])
        classe = 0 if i in class_0 else 1 if i in class_1 else 2 if i in class_2 else 3
        new_samples = create_new_data(data.iloc[i].dropna().to_numpy(),ws, ws-strides[classe])
        new_data.extend(new_samples)
        #classes = classes + [classe] * len(new_samples)
        for j in range(len(new_samples)):
            classes.append(classe)
        print(i)


    """print(len(new_data))
    print(classes.count(0),classes.count(1),classes.count(2),classes.count(3))"""

    """import matplotlib.pyplot as plt
    plt.plot(range(0,len(new_data[112])),new_data[112])"""


    with open(pathOutput, 'wb') as f:
        pickle.dump((new_data,classes), f)