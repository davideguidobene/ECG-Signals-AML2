import pickle

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dropout, Add, MaxPooling1D

from tensorflow.keras.layers import Bidirectional,LSTM

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape

from tensorflow.keras import Model
from tensorflow.keras import Input


#Load data from the pickle file
le_file_path = 'evenet_data.pkl'
with open(le_file_path, 'rb') as file:
    loaded_data = pickle.load(file)

new_data, classes = loaded_data

def residual_block(x, filters, kernel_size, dropout_rate):
    # Convolutional layer
    conv = Conv1D(filters=filters,kernel_size= kernel_size,strides=2, 
                            activation=None,padding='same')(x)
    bn = BatchNormalization()(conv)
    relu = ReLU()(bn)
    dropout = Dropout(dropout_rate)(relu)

    fin = Conv1D(filters=filters,kernel_size= kernel_size, 
                            activation=None,padding='same')(dropout)

    x = MaxPooling1D(pool_size=2)(x)
    
    # Residual connection
    residual = Add()([x, fin])
    return residual

def residual_block2(x, filters, kernel_size, dropout_rate):
    
    #print(x.shape)
    X = BatchNormalization()(x)
    #print(X.shape)
    X = ReLU()(X)
    #print(X.shape)
    X = Dropout(dropout_rate)(X)
    #print(X.shape)
    X = Conv1D(filters=filters,kernel_size= kernel_size,strides=2, 
                            activation=None,padding='same')(X)
    #print(X.shape)
    X = BatchNormalization()(X)
    #print(X.shape)
    X = ReLU()(X)
    #print(X.shape)
    X = Dropout(dropout_rate)(X)
    #print(X.shape)
    X = Conv1D(filters=filters,kernel_size= kernel_size, 
                            activation=None,padding='same')(X)
    #print(X.shape)
    x = MaxPooling1D(pool_size=2)(x)
    #print(x.shape)

    # Residual connection
    residual = Add()([x, X])
    return residual

# MODEL ################################################################################
inputs = tf.keras.Input(shape=(1024,1))
Conv = Conv1D(filters=64,kernel_size= 16,strides=2,activation=None,padding='same')(inputs)
bn = BatchNormalization()(Conv)
relu = ReLU()(bn)
#print(relu.shape)
#model = Model(inputs,relu)
res = residual_block(relu,64,16,0.2)
#print(res.shape)
#model = Model(inputs,res)
num_residual_layers = 4

for i in range(num_residual_layers):
    res = residual_block2(res,64,16,0.2)
    #print(res.shape)

last_layer = BatchNormalization()(res)
last_layer = ReLU() (last_layer)
last_layer = Bidirectional(LSTM(64, return_sequences=True))(last_layer)
#print(last_layer.shape)
last_layer = Dropout(0.2)(last_layer)

features = Dense(1, activation='sigmoid')(last_layer)
features = (Reshape((16,), input_shape=(1, 16, 1)))(features)
#print(features.shape)

logits = Dense(units=4, activation='softmax') (features)
#print(logits.shape)
model = Model(inputs,logits)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()


# TRAINING #############################################################################
x_data = new_data
y_data = classes
threshold = round(len(x_data)*0.8)
print(threshold)

print("Shuffling...")
##shuffle
import random
zipped = list(zip(x_data, y_data))
random.shuffle(zipped)
x_data, y_data = zip(*zipped)
#print(tf.shape(X_train), tf.shape(Y_train))
#print(len(X_train), len(Y_train))
print("Shuffling complete")


X_train = x_data[:threshold]
Y_train = y_data[:threshold]
X_train = tf.convert_to_tensor(X_train)
X_train = tf.reshape(X_train, shape=(len(X_train),1024,1)) 
Y_train = tf.convert_to_tensor(Y_train)
Y_train = tf.reshape(Y_train, shape=(len(Y_train),1,1))


X_val = x_data[threshold:]
Y_val = y_data[threshold:]
X_val = tf.convert_to_tensor(X_val)
X_val = tf.reshape(X_val,shape=(len(X_val),1024,1))
Y_val = tf.convert_to_tensor(Y_val)
Y_val = tf.reshape(Y_val,shape=(len(Y_val),1,1))

print("test")
print(tf.shape(X_train), tf.shape(Y_train))
history = model.fit(x=X_train,y=Y_train,epochs=10,batch_size=1,validation_data=(X_val,Y_val))
#print(model.predict(new_data[400]))
#print(classes[400])


# FINAL OPERATIONS #######################################################################
import datetime
model.save("myModel_backup.h5")
name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
model.save(f"myModel_{name}.h5")


"""
# to load a model do this:
loaded_model = tf.keras.models.load_model("myModel.h5")
loaded_model.summary()
second_to_last_name = loaded_model.layers[-2].name

feature_extraction_model = Model(inputs = loaded_model.input,outputs=loaded_model.get_layer(second_to_last_name).output)

input_data = new_data[5]
input_data = tf.reshape(input_data, shape=(1,1024,1))

features = feature_extraction_model(input_data)
features
"""