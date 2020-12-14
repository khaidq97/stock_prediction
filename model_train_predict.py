import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
# from statsmodels.tsa.arima.model import ARIMA
from pmdarima import ARIMA
from sklearn.preprocessing import MinMaxScaler

# hàm chuẩn hóa dữ liệu về (0,1)
def training_data_normalizing(trainData):
    max_price_scaler = max(trainData) + max(trainData)*0.2
    min_price_scaler = max(min(trainData) - min(trainData)*0.2, 0)

    trainData = np.append(trainData, [max_price_scaler, min_price_scaler])
    trainData = trainData.reshape([-1,1])

    sc = MinMaxScaler(feature_range=(0, 1))
    trainData = sc.fit_transform(trainData)
    trainData = trainData[:-2]

    return sc, trainData

# Hàm early stopping
# This callbach will stop the training when there is no improvement in
# the loss for three consecutive epochs
def custom_callback(monitor="val_loss", min_delta=0, patience=50, verbose=1, mode="min"):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor= monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        mode=mode,
        baseline=None,
        restore_best_weights=True
    )
    return callback

# model LSTM
def LSTM_model(data, optimizer='adam', loss='mean_squared_error', val_split=0.2, epochs=30, batch_size=32, verbose=1, timeStep=8):
    sc, data = training_data_normalizing(data)

    X = []
    y = []
    for i in range(timeStep, len(data)):
        X.append(data[i-timeStep:i-1,0])
        y.append(data[i:i+1,0])

    X, y = np.array(X), np.array(y)
    X = X[:,:,np.newaxis]


    model_lstm = Sequential()

    model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],1), stateful=False))
    model_lstm.add(Dropout(0.2))

    # model_lstm.add(LSTM(units=20, return_sequences=True, stateful=False))
    # model_lstm.add(Dropout(0.2))

    # model_lstm.add(LSTM(units=20, return_sequences=True, stateful=False))
    # model_lstm.add(Dropout(0.2))

    model_lstm.add(LSTM(units=20, return_sequences=False, stateful=False))
    model_lstm.add(Dropout(0.2))

    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer=optimizer, loss=loss)

    callback = custom_callback(min_delta=0, patience=35)
    hist_lstm = model_lstm.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=val_split, callbacks=[callback])

    return model_lstm, hist_lstm, sc

# model ARIMA
def ARIMA_model(x, order=(1,1,0), mode="resid"):
    try:
        arima_init = ARIMA(order=order)
        model = arima_init.fit(x)
        if mode == "resid":
            return model.resid()
        if mode == "predict":
            return model.predict(n_periods=1)
        
    except:
        print("Something wrong!!!")


# Model ARIMA + LSTM
def ARIMA_LSTM_model(X, order=(1,1,0), optimizer='adam', loss='mean_squared_error', val_split=0.2, epochs=30, batch_size=32, verbose=1, timeStep=8):
    X_train = np.array([X[i-timeStep:i,0] for i in range(timeStep, len(X))])

    X_resid = []
    y = []
    for i in range(len(X_train)):
        resid = ARIMA_model(X_train[i], order=order, mode='resid')
        X_resid.append(resid[:timeStep-1])
        y.append(resid[timeStep-1])
    X_resid, y = np.array(X_resid), np.array(y)

    X_resid = X_resid[:,:, np.newaxis]

    model_lstm = Sequential()

    model_lstm.add(LSTM(units=100, return_sequences=True, input_shape=(X_resid.shape[1],1), stateful=False))
    model_lstm.add(Dropout(0.2))

    # model_lstm.add(LSTM(units=20, return_sequences=True, stateful=False))
    # model_lstm.add(Dropout(0.2))

    # model_lstm.add(LSTM(units=20, return_sequences=True, stateful=False))
    # model_lstm.add(Dropout(0.2))

    model_lstm.add(LSTM(units=20, return_sequences=False, stateful=False))
    model_lstm.add(Dropout(0.2))

    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer=optimizer, loss=loss)

    callback = custom_callback(min_delta=0, patience=45)
    hist_lstm = model_lstm.fit(X_resid, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=val_split, callbacks=[callback])

    return model_lstm, hist_lstm


def ARIMA_LSTM_predict(model_lstm, data,order=(1,1,0), timeStep=8):
    X = []
    y = []
    for i in range(timeStep, len(data)):
        X.append(data[i-timeStep:i-1,0])
        y.append(data[i,0])

    y_pred = []
    for i in range(len(X)):
        resid = ARIMA_model(X[i],order=order, mode='resid')[np.newaxis,:,np.newaxis]

        y_pred.append(ARIMA_model(X[i],order=order, mode='predict') + model_lstm.predict(resid))

    y_pred = np.array(y_pred).reshape([-1,1])

    return y, y_pred

def ARIMA_single(data, order=(1,1,0), timeStep=8):
    X = []
    y = []
    for i in range(timeStep, len(data)):
        X.append(data[i-timeStep:i,0])
        y.append(data[i,0])

    y_pred = []
    for i in range(len(X)):

        y_pred.append(ARIMA_model(X[i],order=order, mode='predict'))

    y_pred = np.array(y_pred).reshape([-1,1])

    return y, y_pred


def LSTM_predict(model_lstm, data, sc, timeStep=8):
    X = []
    y = []
    for i in range(timeStep, len(data)):
        X.append(data[i-timeStep:i-1,0])
        y.append(data[i:i+1,0])

    X, y = np.array(X), np.array(y)

    X = sc.transform(X)[:,:,np.newaxis]
    
    y_pred = sc.inverse_transform(model_lstm.predict(X))

    return y, y_pred

