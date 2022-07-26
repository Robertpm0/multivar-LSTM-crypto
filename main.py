
import math
from random import randint
import pandas_datareader as web
import pandas as pd
import  numpy  as np
from regex import R
from sklearn.mixture  import GaussianMixture
from sklearn.preprocessing import scale
import sklearn
import sklearn.mixture
from sklearn.preprocessing  import  MinMaxScaler
from pandas import DataFrame
from pandas import concat
import tensorflow 
import keras
from keras.layers import Bidirectional, LSTM, Dropout,  Dense, Activation


import matplotlib.pyplot as plt




df  =  web.DataReader('BTC-USD', 'yahoo')
df.reset_index(inplace=True)
#print(df)
data = df[['Date','High', 'Low', 'Adj Close', 'Volume']]

dete  = pd.DataFrame(data)
training_dataset =  df[['High', 'Low', 'Adj Close', 'Volume']]
training_dataset.drop(
    training_dataset[training_dataset['Volume']==0].index,
    inplace = True)
#print(training_dataset)
training_dataset_ext = training_dataset.copy()
training_dataset_ext['Prediction'] = training_dataset['Adj Close']
nRows =  training_dataset.shape[0]
npDataUnscale = np.array(training_dataset)
npData = np.reshape(npDataUnscale, (nRows,  -1))
scaler = MinMaxScaler()
npDataScale = scaler.fit_transform(npDataUnscale)                                                                                

scalerPred = MinMaxScaler()
dfClose = pd.DataFrame(training_dataset_ext['Adj Close'])
dfCloseScale = scalerPred.fit_transform(dfClose)

seqLen =  50

indexClose = training_dataset.columns.get_loc('Adj Close')

trainLen = math.ceil(npDataScale.shape[0] * 0.85)
print((trainLen))

trainData = npDataScale[0:trainLen, :]
testData = npDataScale[trainLen - seqLen:, :]
print(trainData)
print(testData)

def preprocessTrain(sequenceLen, data):
    x, y = [],[]
    dataLen = data.shape[0]
    for i  in range(sequenceLen, dataLen-3):
        x.append(data[i-sequenceLen:i,:])
        y.append(data[i+3, indexClose])
        
    
    x = np.array(x)
    y = np.array(y)
    return x, y

def preprocessTest(sequenceLen, data):
    x, y = [],[]
    dataLen = data.shape[0]
    for i  in range(sequenceLen, dataLen):
        x.append(data[i-sequenceLen:i,:])
        y.append(data[i, indexClose])
    x = np.array(x)
    y = np.array(y)
    return x, y

xTrain, yTrain  = preprocessTest(seqLen, trainData)
xTest, yTest = preprocessTest(seqLen, testData)


#print(training_dataset)


dropOut = 0.2
#windowSize = seqLen -1 ###mess aroun with thi s1 see what it does to model perform.....
windowSize = xTrain.shape[1] * xTrain.shape[2]

model = keras.Sequential()

model.add(Bidirectional(LSTM(windowSize, return_sequences=True, input_shape=(xTrain.shape[1], xTrain.shape[2]))))
model.add(Dropout(rate=dropOut))

model.add(Bidirectional(LSTM(windowSize, return_sequences=True)))
model.add(Dropout(rate=dropOut))
model.add(Bidirectional(LSTM(windowSize, return_sequences=False)))
model.add(Dense(units=1))
model.add(Activation('relu'))
model.compile(loss='mae', optimizer='adam')

model.fit(xTrain,yTrain, epochs=25, batch_size=32, shuffle=False)

yHat = model.predict(xTest)



prices =  scalerPred.inverse_transform(yHat)
price = scalerPred.inverse_transform(yTest.reshape(-1,1))

print(price)

#reframed = series_to_supervised(scaledValues,  1,1)
#scaledValues = scaledValues.reshape(-1,1)
#print(scaledValues)
#print(reframed)
