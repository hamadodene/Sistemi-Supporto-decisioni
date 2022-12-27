import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import math


# Data Upload
df = pd.read_csv("serieFcast2021.csv")

# Preprocessing
df = df.fillna(df.interpolate())  # filling the NaN values
amount = df["amount"]  # array of amount data
logdata = np.log(amount)  # Log transform
data = pd.Series(logdata)  # Convert to pandas series

df["amount"].plot()
plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Data'])
plt.title('Raw data')
plt.show()

data.plot()
plt.xlabel('time')
plt.legend(['Log data'])
plt.ylabel('log amounts')
plt.title('Log data')

plt.show()

# ACF plot
sm.graphics.tsa.plot_acf(data.values, lags=24)
plt.show()

# train and test set
cutpoint = int(0.7 * len(data))
train = data[:cutpoint]
test = data[cutpoint:]

train_raw = amount[:cutpoint]
test_raw = amount[cutpoint:]


# AUTORIMA WITH TRAIN DATA

model = pm.auto_arima(train.values, start_p=1, start_q=1,
                      test='adf', max_p=3, max_q=3, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)  # False full grid
print(model.summary())
morder = model.order
morder = model.order;
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order
print("Sarimax seasonal order {0}".format(mseasorder))

# predictions and forecast
fitted = model.fit(train)
ypred = fitted.predict_in_sample()  # prediction (in-sample)
yfore = fitted.predict(n_periods=12)  # forecast (out-of-sample)

plt.plot(train)
plt.plot(test)

plt.plot([None for i in ypred] + [x for x in yfore], color="red")  # forecast
plt.plot([None for x in range(12)] + [x for x in ypred[12:]], color="orange")  # prediction
plt.xlabel('time')
plt.ylabel('amount')
plt.legend(['Train', 'Test', 'Forecast', 'Prediction'])
plt.show()

# reconstruction
yplog = pd.Series(ypred)
expdata = np.exp(yplog)  # unlog
expfore = np.exp(yfore)

plt.plot(train_raw)
plt.plot(test_raw)
plt.plot([None for x in range(36)] + [x for x in expdata[36:]], color="red")
plt.plot([None for x in expdata] + [x for x in expfore], color="orange")  # prediction
plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Train', 'Test', 'Forecast', 'Prediction'])
plt.show()

# ------------------ using statsmodels’ SARIMAX, morder derived in auto_arima
sarima_model = SARIMAX(train.values, order=morder, seasonal_order=mseasorder)
sfit = sarima_model.fit()
sfit.plot_diagnostics()
plt.show()

ypred = sfit.predict(start=0, end=len(train))
yfore = sfit.get_forecast(steps=24)
expdata = np.exp(ypred)  # unclog
expfore = np.exp(yfore.predicted_mean)

plt.plot(train_raw)
plt.plot(test_raw)
plt.plot(expdata)
plt.plot([None for x in expdata] + [x for x in expfore])
plt.legend(['Train', 'Test', 'Prediction', 'Forecast'])
plt.show()

# AUTOARIMA FULL DATASET
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

# predictions and forecasts
fitted = model.fit(logdata)
ypred = fitted.predict_in_sample()  # prediction (in-sample)
yfore = fitted.predict(n_periods=24)  # forecast (out-of-sample)

plt.plot(logdata, color="green")  # Dati di train
plt.plot([None for x in ypred] + [x for x in yfore], color="red")  # Predict
plt.plot([None for x in range(24)] + [x for x in ypred[24:]], color="orange")  # Forecast

plt.xlabel('time')
plt.ylabel('log amounts')
plt.legend(['Train', 'Prediction', 'Forecast'])
plt.title('AUTOARIMA FULL DATASET')

plt.show()

# # # SARIMAX WITH FULL DATASET # # #

sarima_model = SARIMAX(logdata, order=morder, seasonal_order=mseasorder)
sfit = sarima_model.fit()
sfit.plot_diagnostics()
plt.show()

ypred = sfit.predict(start=0, end=len(logdata))
yfore = sfit.get_forecast(steps=24)
expdata = np.exp(ypred)  # unlog
expfore = np.exp(yfore.predicted_mean)

# MSE & RMSE on full data set
mse_train = mean_squared_error(expdata[1:], df.amount)
rmse_train = math.sqrt(mse_train)
print('Full dataset score - SARIMAX | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_train, rmse_train))

plt.plot([None for x in expdata] + [x for x in expfore], color="red")
plt.plot(df.amount, color="green")
plt.plot(expdata, color="orange")

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Forecast', 'Raw train data', 'Prediction'])
plt.title('SARIMAX with entire reconstructed data set')

plt.show()