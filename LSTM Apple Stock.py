import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv('AAPL Stock.csv')

print(df)

df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)

plt.plot(df['Date'],df['Close'])
plt.show()
df.index = df['Date']

print(df)

df.drop(['Date','Open','High','Low','Adj Close','Volume'],axis=1,inplace=True)

print(df)

df = np.array(df).reshape(-1,1)

print(df)

scaler = StandardScaler()

df_transform = scaler.fit_transform(df)

print(df_transform)

train = df_transform[:1530]

test = df_transform[1530:]

train = np.array(train).reshape(-1,1)

test = np.array(test).reshape(-1,1)


def dataset(dataset,timestep=5):
    dataX,dataY = [],[]
    
    for i in range(0,len(dataset) - timestep):
        a = dataset[i : i + timestep]
        
        dataX.append(a)
        
        b = dataset[i + timestep]
        dataY.append(b)
        
    return np.array(dataX),np.array(dataY)

x_train,y_train = dataset(train)

x_test,y_test = dataset(test)

print(x_train)
print(y_train)

model = keras.Sequential([keras.layers.LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)),
                          keras.layers.LSTM(50),
                          keras.layers.Dense(25,activation='relu'),
                          keras.layers.Dense(1)
                         
                          ])




model.compile(optimizer=keras.optimizers.Adam(0.007),loss='mse',metrics=['mae'])

hist = model.fit(x_train,y_train,validation_split=0.2,epochs=150,verbose=1,callbacks=keras.callbacks.EarlyStopping('val_loss',0.10,patience=10))

predict = model.predict(x_test)

predict_inv = scaler.inverse_transform(predict)

y_test_inv = scaler.inverse_transform(y_test)


plt.plot(list(range(1530,(len(df)-5))),predict_inv)
plt.plot(df)



plt.legend(['Predicted','Actual'])

plt.show()


plt.plot(predict_inv)
plt.plot(y_test_inv)

plt.legend(['Predicted','Actual'])

plt.show()

print(y_test_inv[:20])
print(predict_inv[:20])

def test_dataset(dataset,time_step=5):
    data_test = []
    
    for i in range(0,len(dataset) - time_step):
        a = dataset[i:i + time_step]
        data_test.append(a)
        
    return np.array(data_test)
        

future_test = np.array([
149.790466,
147.711563,
149.671112,
151.779831,
152.177719
]).reshape(-1,1)

future_test = scaler.transform(future_test)

future_test = future_test.reshape(1,5,1)

# future_test_dataset = test_dataset(future_test,1)

print(x_test.shape)
print(future_test.shape)
# print(future_test_dataset)

future_test_pred = model.predict(future_test)

future_pre = scaler.inverse_transform(future_test_pred)

print(future_pre)


plt.plot(hist.history['loss'])

plt.show()

