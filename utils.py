import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
# from pmdarima import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


def plot_actual_predict(y, y_pred):
    plt.figure(figsize=(20,10))
    plt.plot(y, color='red', label='Actual Stock Price' )
    plt.plot(y_pred, color='green', label='Predict Stock Price')
    plt.title('Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def plot_cumsum(df, ticker):
	dr = df.cumsum()
	dr.plot()
	plt.title(f'The {ticker} stock')
	plt.show()

def autocorrelation(df, ticker):
	plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

	# Plot
	fig, axes = plt.subplots(1, 7, figsize=(12,2), sharex=True, sharey=True, dpi=100)
	for i, ax in enumerate(axes.flatten()[:7]):
		lag_plot(df.Mean, lag=i+1, ax=ax, c='green')
		ax.set_title('Lag ' + str(i+1))

	fig.suptitle(f'The {ticker} stock', y=1.15)    
	plt.show()

def check_stationarity(df):
    ts_data = df.copy()
    ts_data.dropna(inplace=True)

    # Rolling statistics
    roll_mean = ts_data.rolling(30).mean()
    roll_std = ts_data.rolling(5).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(20,10))
    plt.subplot(211)
    plt.plot(ts_data, color='black', label='Original Data')
    plt.plot(roll_mean, color='red', label='Rolling Mean(30 days)')
    plt.legend()
    plt.subplot(212)
    plt.plot(roll_std, color='green', label='Rolling Std Dev(5 days)')
    plt.legend()
    
    # Dickey-Fuller test
    print('Dickey-Fuller test results\n')
    df_test = adfuller(ts_data, regresults=False)
    test_result = pd.Series(df_test[0:4], index=['Test Statistic','p-value','# of lags','# of obs'])
    print(test_result)
    for k,v in df_test[4].items():
        print('Critical value at %s: %1.5f' %(k,v))

def autocorrelation_partialAutocorrection(df):
	df_acf = acf(df.Mean)
	df_pacf = pacf(df.Mean)

	fig1 = plt.figure(figsize=(20,10))
	ax1 = fig1.add_subplot(211)
	#significance limit region
	fig1 = sm.graphics.tsa.plot_acf(df_acf, ax=ax1)

	ax2 = fig1.add_subplot(212)
	#significance limit region
	fig1 = sm.graphics.tsa.plot_pacf(df_pacf, ax=ax2)


# ROOT MEAN SQUARE ERROR 
def evaluate_RMSE(y, y_pred):

    RMSE = mean_squared_error(y, y_pred, squared=False)

    MAE = mean_absolute_error(y, y_pred)
    
    return RMSE, MAE


def plot_history(history):
    plt.title('Loss / Mean Squared Error')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


#############################################################################
# use grid search to choice parameters for ARINA model#######################
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = np.sqrt(mean_squared_error(test, predictions))
	return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

	return best_cfg


def ARIMA_search_params(data, p_values=range(0, 3), d_values=range(0, 3), q_values=range(0, 3)):

    return evaluate_models(data, p_values, d_values, q_values)


# CHECK TREND
def check_trend(y):
    labels = []
    for i in range(len(y)-1):
        trend = y[i] - y[i+1]
        if trend >= 0:
            labels.append(1)
        elif trend < 0:
            labels.append(-1)
    
    return labels

# EVALUATE ACCURACY. PRECISION SCORE ACCORDING TO THE TREND
def evaluate_trend(y, y_pred):

	y_trend = check_trend(y)
	y_trend_pred = check_trend(y_pred)
	print("Accuracy according to the trend:", round(accuracy_score(y_trend, y_trend_pred), 4))
	print("Precision  according to the trend:", round(precision_score(y_trend, y_trend_pred), 4))