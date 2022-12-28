import math

import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
from keras.layers import Dense  # pip install tensorflow (as administrator)
from keras.models import Sequential  # pip install keras
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

# Data Upload
df = pd.read_csv("serieFcast2021.csv")
df.amount.plot()
plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Data'])
plt.title('Raw data')
plt.show()

# Preprocessing
df = df.fillna(df.interpolate())  # remove the NaN values
amount = df["amount"]  # array of amount data
logdata = np.log(amount)  # Log transform
data_series = pd.Series(logdata)  # Convert to pandas series

df.amount.plot()
plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Data'])
plt.title('Raw data without NaN')
plt.show()

data_series.plot()
plt.xlabel('time')
plt.legend(['Log data'])
plt.ylabel('log amounts')
plt.title('Log data')
plt.show()

# ACF
diffdata = data_series.diff()
diffdata[0] = data_series[0]  # reset 1st elem
acfdata = acf(diffdata, adjusted=True, nlags=24)
plt.bar(np.arange(len(acfdata)), acfdata)
plt.title("Auto-correlation with bar")
plt.show()

# otherwise
sm.graphics.tsa.plot_acf(data_series.values, lags=24)
plt.show()

# train and test set
cutpoint = int(0.7 * len(data_series))
train = data_series[:cutpoint]
test = data_series[cutpoint:]

train_raw = amount[:cutpoint]
test_raw = amount[cutpoint:]

# -------------------- STATISTIC MODELS -----------------------
# Auto - ARINA WITH FULL DATA
model = pm.auto_arima(logdata, start_p=1, start_q=1,
                      test='adf', max_p=3, max_q=3, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)  # False full grid

print(model.summary())
morder = model.order
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order
print("Sarimax seasonal order {0}".format(mseasorder))

fitted = model.fit(logdata)
yfore = fitted.predict(n_periods=24)  # forecast (out-of-sample)
ypred = fitted.predict_in_sample()  # prediction (in-sample)

plt.plot(logdata)  # Dati di train
plt.plot([None for i in ypred] + [x for x in yfore])  # prediction
plt.plot([None for x in range(12)] + [x for x in ypred[12:]])  # forecast

plt.xlabel('time')
plt.ylabel('amount')
plt.legend(['Train', 'Prediction', 'Forecast'])
plt.title('AUTOARIMA FULL DATASET')
plt.show()

# Reconstruction
yplog = pd.Series(ypred)
expdata = np.exp(yplog)  # unlog
expfore = np.exp(yfore)

plt.plot(amount)
plt.plot([None for x in range(24)] + [x for x in expdata[24:]], color="orange")  # Prediction
plt.plot([None for x in expdata] + [x for x in expfore], color="red")  # forecast

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Train', 'Prediction', 'Forecast'])
plt.title('AUTOARIMA RECONSTRUCTED DATASET')

plt.show()

# SARIMAX FULL DATASET
sarima_model = SARIMAX(logdata, order=morder, seasonal_order=mseasorder)
sfit = sarima_model.fit()
sfit.plot_diagnostics()
plt.title("SARIMAX diagnostics")
plt.show()

ypred = sfit.predict(start=0, end=len(logdata))
yfore = sfit.get_forecast(steps=24)
expdata = np.exp(ypred)  # unlog
expfore = np.exp(yfore.predicted_mean)

# MSE & RMSE on train set
mse_train = mean_squared_error(expdata[1:], amount)
rmse_train = math.sqrt(mse_train)
print('Train Score - SARIMAX | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_train, rmse_train))

plt.plot(expdata)
plt.plot(df.amount)
plt.plot([None for x in expdata] + [x for x in expfore])  # Forecast
plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Prediction', 'Raw data', 'Forecast'])
plt.title('SARIMAX FULL DATASET')
plt.show()

# Auto - ARINA WITH TRAIN DATA
model = pm.auto_arima(train.values, start_p=1, start_q=1,
                      test='adf', max_p=3, max_q=3, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)  # False full grid

print(model.summary())
morder = model.order
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order
print("Sarimax seasonal order {0}".format(mseasorder))

fitted = model.fit(train)
yfore = fitted.predict(n_periods=24)  # forecast (out-of-sample)
ypred = fitted.predict_in_sample()  # prediction (in-sample)

plt.plot(train)
plt.plot(test)
plt.plot([None for i in ypred] + [x for x in yfore])  # prediction
plt.plot([None for x in range(12)] + [x for x in ypred[12:]])  # forecast
plt.xlabel('time')
plt.ylabel('amount')
plt.legend(['Train', 'Test', 'Prediction', 'Forecast'])
plt.title('AUTOARIMA WITH TRAIN DATA')
plt.show()

# Reconstruction
yplog = pd.Series(ypred)
expdata = np.exp(yplog)  # unlog
expfore = np.exp(yfore)

plt.plot(amount)
plt.plot([None for x in range(24)] + [x for x in expdata[24:]], color="orange")  # Prediction
plt.plot([None for x in expdata] + [x for x in expfore], color="red")  # forecast

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Train', 'Prediction', 'Forecast'])
plt.title('AUTOARIMA RECONSTRUCTED DATASET')
plt.show()

# Preprocessed data
# SARIMAX with train data
sarima_model = SARIMAX(train, order=morder, seasonal_order=mseasorder)
sfit = sarima_model.fit()
sfit.plot_diagnostics()
plt.title("SARIMAX diagnostics")
plt.show()

ypred = sfit.predict(start=0, end=len(train))
yfore = sfit.get_forecast(steps=36)
expdata = np.exp(ypred)  # unlog
expfore = np.exp(yfore.predicted_mean)

# MSE & RMSE on train set
mse_train = mean_squared_error(expdata[1:], train_raw)
rmse_train = math.sqrt(mse_train)
print('Train Score - SARIMAX | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_train, rmse_train))

plt.plot(train_raw)  # raw train data
plt.plot(test_raw)
plt.plot(expdata)
plt.plot([None for x in expdata] + [x for x in expfore])  # Forecast

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Train', 'Test', 'Prediction', 'Forecast'])
plt.title('SARIMAX TRAIN DATASET')
plt.show()


# -------------------- PREDICTIVE NEURALS METHOD  -----------------------

# from series of values to windows matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


np.random.seed(550)  # for reproducibility
df = pd.read_csv("serieFcast2021.csv", usecols=[1], names=["amount"], header=0).interpolate()

dataset = df.values  # time series values
dataset = dataset.astype('float32')  # needed for MLP input

# split into train and test sets
train_size = int(len(dataset) - 12)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Len train={0}, len test={1}".format(len(train), len(test)))

# sliding window matrices (look_back = window width); dim = n - look_back - 1
look_back = 2
testdata = np.concatenate((train[-look_back:], test))
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(testdata, look_back)

# Multilayer Perceptron model
loss_function = 'mean_squared_error'
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))  # 8 hidden neurons
model.add(Dense(1))  # 1 output neuron
model.compile(loss=loss_function, optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore,
                                                           math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))
# generate predictions for training and forecast for plotting
trainPredict = model.predict(trainX)
testForecast = model.predict(testX)

# predict for 24 more periods
result = testForecast
for n in np.linspace(0, 23, 24):
    np.append(result, model.predict(np.asarray([[result[-2:, 0][0], result[-1:, 0][0]]],
                                               dtype="float32")))

plt.plot(dataset)
plt.plot(np.concatenate((np.full(look_back - 1, np.nan), trainPredict[:, 0])))
plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), testForecast[:, 0])))
plt.plot(np.concatenate((np.full(len(train) + 11, np.nan), result[:, 0])))

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Dataset', 'Train Predict', 'Test Forecast', 'Prediction'])
plt.title('Multilayer Perceptron model')
plt.show()

mse_mlp_full = mean_absolute_error(testY, testForecast)
rmse_mlp_full = math.sqrt(mse_mlp_full)
print('Full dataset score | MLP | MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(mse_mlp_full, rmse_mlp_full))

# -------------------- MACHINE LEARNING METHOD -----------------------
mdl = rf = RandomForestRegressor(n_estimators=500)
mdl.fit(trainX, trainY)
trainPredict = mdl.predict(trainX)
testForecast = mdl.predict(testX)

plt.plot(dataset)
plt.plot(np.concatenate((np.full(look_back - 1, np.nan), trainPredict)))
plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), testForecast)))
plt.title('MACHINE LEARNING METHOD')
plt.legend(['Dataset', 'Train Predict', 'Test Forecast'])
plt.show()

print("MSE={}".format(mean_absolute_error(testY, testForecast)))

