"""
Author - Group 62
"""
import datetime as dt 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt
import statsmodels.api as smapi
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.optimize import brute
import warnings
import time

from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler

import holtwinters as hw 
from dateutil.relativedelta import relativedelta 

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

"""
Seasonal Naive Bayes - This function plots the residual error density plot
"""

def SNB_residual_plot(test_list,predict_list):
    error = [test_list[i]-predict_list[i] for i in range(len(test_list))]
    residual_error = pd.DataFrame(error)
    print("Residual Error")
    print("--------------")
    print(residual_error.describe())
    # plot Residual error distribution
    fig = plt.figure(figsize=(6, 3))
    residual_error.plot(kind='kde', ax=plt.gca())
    plt.title("Residual Error for Seasonal Naive Bayes")
    plt.show(block=False)
    fig.savefig("Seasonal NB - Residual Error for Seasonal NB.pdf")

"""
ARIMA - This function decompose the time series and plots the seasonality, trend and residual.
This function returns the seasonal adjust data for the input training data.
"""

def ARIMA_series_decompose(period, train):
    decomp_obj = smapi.tsa.seasonal_decompose(train)
    season_adj = train - decomp_obj.seasonal
    fig = plt.figure(figsize=(10, 5))
    plt.plot(decomp_obj.seasonal)
    plt.title("Seasonal Decomposition -Seasonality of training dataset" + period)
    plt.xlabel("Months")
    plt.show(block=False)
    fig.savefig("ARIMA - Seasonal Decomposition -Seasonality of training dataset "+period+".pdf")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(decomp_obj.trend)
    plt.title("Seasonal Decomposition - Trend of training dataset" + period)
    plt.xlabel("Months")
    plt.show(block=False)
    fig.savefig("ARIMA - Seasonal Decomposition -Trend of training dataset "+period+".pdf")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(decomp_obj.resid)
    plt.title("Seasonal Decomposition - Residual of training dataset" + period)
    plt.xlabel("Months")
    plt.show(block=False)
    fig.savefig("ARIMA - Seasonal Decomposition - Residual of training dataset "+period+".pdf")
    return season_adj


"""
ARIMA - This function intakes the seasonal adjusted data and performs first order differencing.
This function plots the ACF and PACF plots for the differenced data.
"""

def ARIMA_ACF_PACF(period, seasonal_adj):
    diffed = seasonal_adj.diff(1)[1:]
    fig = plt.figure(figsize=(10, 5))
    plt.plot(diffed)
    plt.title("First Order Differencing - " + period)
    plt.xlabel("Months")
    plt.show(block=False)
    fig.savefig("ARIMA - First Order Differencing " + period + ".pdf")
    fig = plt.figure(figsize=(6,6))
    plt.subplot(211)
    smt.graphics.plot_pacf(diffed, lags=24,ax=plt.gca())
    plt.subplot(212)
    smt.graphics.plot_acf(diffed, lags=24,ax=plt.gca())
    plt.show(block=False)
    fig.savefig("ARIMA - ACF and PACF plots on differenced series "+period+".pdf")

"""
ARIMA - This function calculates the optimal seasonal ARIMA order using brute force method
"""

def object_func(arima_order, data):
    order = [int(x) for x in arima_order]
    try:
        fit = smapi.tsa.statespace.SARIMAX(data, trend='n', order=arima_order[:3], seasonal_order=order[3:]).fit(disp=0)
        predict = fit.predict(disp=0)
        rmse = mt.sqrt(mean_squared_error(data, predict))
        return rmse
    except:
        return np.inf

"""
ARIMA - This function calculates the test RMSE and MAE scores for the optimal ARIMA order passed as argument.
"""

def ARIMA_calc_train_test_RMSE(period, optimal_converted):
    seasonal_model = smapi.tsa.statespace.SARIMAX(train, trend='n', order=optimal_converted[:3],
                                                  seasonal_order=optimal_converted[3:])
    results = seasonal_model.fit(disp=0)
    seasonal_forecast = results.predict(end="2017-3-1")
    predicted = seasonal_forecast[seasonal_forecast.index >= '2016-04-01']
    rmse = mt.sqrt(mean_squared_error(test, predicted))
    print('')
    print(period)
    print('---------------------------')
    print('Optimal ARIMA Order', optimal_converted)
    print('ARIMA - Test RMSE:', " is: ", round(rmse, 2))
    mape = np.mean(np.abs((test - predicted) / test)) * 100
    print('ARIMA - Test MAPE:', mape)
    print('')
    return predicted

"""
ARIMA - This function plots the residual error density plot and the ACF and PACF plots
"""

def ARIMA_residual_plot(period, test_list,predict_list):
    error = [test_list[i]-predict_list[i] for i in range(len(test_list))]
    residual_error = pd.DataFrame(error)
    print("Residual Error")
    print("--------------")
    print(residual_error.describe())
    # plot Residual error distribution
    fig = plt.figure(figsize=(6, 3))
    residual_error.plot(kind='kde', ax=plt.gca())
    plt.title("Residual Error for SARIMA" + period)
    plt.show(block=False)
    fig.savefig("ARIMA - Residual Error for SARIMA"+period+".pdf")
    #plot ACF and PACF of residual
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(211)
    smt.graphics.plot_acf(residual_error, ax=plt.gca())
    plt.subplot(212)
    smt.graphics.plot_pacf(residual_error, ax=plt.gca())
    plt.show(block=False)
    fig.savefig("ARIMA - Residual ACF and PACF plots"+period+".pdf")

"""
HW Function to caculate the MAPE
"""

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

"""
Neural Network - This function will generate and plotting Neural Network prediction 
"""
def generate_LSTM(period, data):
    # To make sure we get the same result
    np.random.seed(1)
    # scaling (preprocessing)
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data.reshape(-1,1))
    # creating training and testing features
    time_window = 12
    Xall, Yall = [], []
    for i in range(time_window, len(data)):
        Xall.append(data[i-time_window:i, 0])
        Yall.append(data[i, 0])
    # Convert to array
    Xall = np.array(Xall)
    Yall = np.array(Yall)
    # Split train and test, test size 12 months
    test_size = 12
    train_size = len(Xall) - test_size
    
    Xtrain = Xall[:train_size, :]
    Ytrain = Yall[:train_size]
    
    Xtest = Xall[:test_size, :]
    Ytest = Yall[:test_size]
    
    # Covert to 3D
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], time_window, 1))  
    Xtest = np.reshape(Xtest, (Xtest.shape[0], time_window, 1))  

    # Recurrent Neural Network is suitable for time-series model
    model = Sequential()
    # Many to one model
    model.add(LSTM(input_shape=(None,1), units=50, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="rmsprop")
    # Fitting the model
    model.fit(Xtrain, Ytrain, batch_size=20, nb_epoch=20, validation_split=0.1)
    # Predict the whole dataset
    allPredict = model.predict(np.reshape(Xall, (len(Xall),len(Xall[0]),1)))
    allPredict = scaler.inverse_transform(allPredict)
    allPredictPlot = np.empty_like(data.reshape(-1,1))
    allPredictPlot[:, :] = np.nan
    allPredictPlot[time_window:, :] = allPredict
    # Plot the graph of original and predicted dataset with one-step prediction
    fig = plt.figure(figsize=(10, 8))
    plt.plot(scaler.inverse_transform(data), label='True Data')
    plt.plot(allPredictPlot, label='One-Step Prediction') 
    plt.title("RNN -  One step prediction - Training dataset " + period)
    plt.xlabel("Months")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()
    fig.savefig("RNN - One Step prediction " + period + ".pdf")
    # Inverse transformation for Training and Test 
    Ytrain = scaler.inverse_transform([Ytrain])
    Ytest = scaler.inverse_transform([Ytest])
    # Estimate training data RMSE and MAPE 
    trainScore = np.sqrt(mean_squared_error(Ytrain[0], allPredict[:train_size,0]))
    trainMapeScore = np.mean(np.abs((Ytrain[0] - allPredict[:train_size,0]) / Ytrain[0])) * 100
    print("For units = 50, batch size=20, epoch=20, validation split =0.1")
    print('Training Data RMSE: {0:.2f}'.format(trainScore))
    print("Training Data MAPE : {0:.2f}".format(trainMapeScore))
    # Estimate testing data RMSE and MAPE 
    testScore = np.sqrt(mean_squared_error(Ytest[0], allPredict[-test_size:,0]))
    testMapeScore = np.mean(np.abs(Ytest[0], allPredict[-test_size:,0]) / Ytest[0]) * 100
    
    print('Test Score: %.2f RMSE' % (testScore))
    print('Test Score: %.2f MAPE' % (testMapeScore))   

    # Dynamic Prediction
    dynamic_prediction = np.copy(data[:len(data) - test_size])
    for i in range(len(data) - test_size, len(data)):
        last_feature = np.reshape(dynamic_prediction[i-time_window:i], (1,time_window,1))
        next_pred = model.predict(last_feature)
        dynamic_prediction = np.append(dynamic_prediction, next_pred)
    
    dynamic_prediction = dynamic_prediction.reshape(-1,1)
    dynamic_prediction = scaler.inverse_transform(dynamic_prediction)
    # Plot the dynamoc prediction and test data
    fig = plt.figure(figsize=(10, 8))
    plt.plot(scaler.inverse_transform(data[:len(data) - test_size]), label='Training Data')
    plt.plot(np.arange(len(data) - test_size, len(data), 1), scaler.inverse_transform(data[-test_size:]), label='Testing Data')
    plt.plot(np.arange(len(data) - test_size, len(data), 1), dynamic_prediction[-test_size:], label='Out of Sample Prediction') 
    plt.title("RNN - Dynamic Prediction " + period)
    plt.xlabel("Months")
    plt.ylabel("Sales")
    plt.legend(loc="upper left")
    plt.show(block=False)
    fig.savefig("RNN - Dynamic Prediction - Training dataset "+period+".pdf")
    # Estimate dynamic test data RMSE and MAPE
    testScore = np.sqrt(mean_squared_error(Ytest[0], dynamic_prediction[-test_size:]))
    testMapeScore = np.mean(np.abs(Ytest[0] - dynamic_prediction[-test_size:]) / Ytest[0]) * 100
    
    print("Dynamic Forecast RMSE : {0:.2f}".format(testScore))
    print("Dynamic Forecast MAPE : {0:.2f}".format(testMapeScore))
    
     
"""
PROGRAM STARTS - We analyse the dataset by plotting the original graph
"""

# Start of Program
start_time = time.time()
warnings.filterwarnings("ignore")

# Read excel file with pandas
cars_xls = pd.ExcelFile("carsalesbystate.xlsx")
cars_df = cars_xls.parse('Data1')
# Column 35 consists of total number of cars sale in Australia
# The first 9 rows
car_sales = cars_df.iloc[9:, 35]
# Convert car_sales to numeric
car_sales = pd.to_numeric(car_sales)
# Plot original series
fig = plt.figure(figsize=(10,6))  # Prepare a figure to draw the time series
plt.plot(car_sales)
plt.title("Original Series - Car sales in Australia from January 1994 to March 2017")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show(block=False)
fig.savefig("ARIMA - Original Series.pdf")

# Generate log value for comparison
car_sales_log = np.log(car_sales)
# Plot log car sales from Jan 2000 to March 2016
fig = plt.figure(figsize=(10,6))
plt.plot(car_sales_log)
plt.title("Log Series - Car sales in log scale in Australia from January 1994 to March 2017")
plt.xlabel("Months")
plt.ylabel("Log - Sales")
plt.show(block=False)
fig.savefig("ARIMA - Log Series.pdf")

months = car_sales.index.strftime('%Y-%m')
# Split series to train and test. Create a list of train series with different starting periods.
trains = []
train_01 = car_sales[(car_sales.index < '2016-04-01') & (car_sales.index >= '1994-01-01')]
trains.append(('Period - Jan 1994 - Mar 2016', train_01))
train_02 = car_sales[(car_sales.index < '2016-04-01') & (car_sales.index >= '2000-01-01')]
trains.append(('Period - Jan 2000 - Mar 2016', train_02))
train_03 = car_sales[(car_sales.index < '2016-04-01') & (car_sales.index >= '2009-01-01')]
trains.append(('Period - Jan 2009 - Mar 2016', train_03))
test = car_sales[car_sales.index >= '2016-04-01']
##################################################################################################################
#                                       NAIVE FORECAST                                                           #
##################################################################################################################
print("Naive Forecast")
print("--------------")
actual_value = [x for x in train_01]
predicted_value = []
for i in range(len(test)):
    predicted_value.append(actual_value[-1])
    actual_value.append(test[i])
rmse = mt.sqrt(mean_squared_error(test, predicted_value))
print('Test data RMSE:', round(rmse, 2))
print('')

# Plot Naive Forecast
fig = plt.figure(figsize=(10, 6))
plt.plot(test, color='blue', label="Original series")
plt.plot(test.index, predicted_value, color='red', label="Naive Forecast, RMSE = " + str(round(rmse, 2)))
plt.title("Naive Forecast Vs Original Observation (Apr 2016 - Mar 2017)")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.legend(loc="upper right")
fig.autofmt_xdate(rotation=90)
plt.show(block=False)
fig.savefig("Naive Forecast Vs Original Observation.pdf")

##################################################################################################################
#                                   SEASONAL NAIVE FORECAST                                                      #
##################################################################################################################
print("Seasonal Naive Forecast")
print("-----------------------")
actual_value = [x for x in train_01]
predicted_value = []
for i in range(len(test)):
    predicted_value.append(actual_value[-12])
    actual_value.append(test[i])
rmse = mt.sqrt(mean_squared_error(test, predicted_value))
print('Test data RMSE:', round(rmse, 2))
print("Test data MAPE: {0:.2f}".format(mean_absolute_percentage_error(test, predicted_value)))


SNB_residual_plot(test,predicted_value)

# Plot Seasonal Naive Forecast
fig = plt.figure(figsize=(10, 6))
plt.plot(test, color='blue', label="Original series")
plt.plot(test.index, predicted_value, color='red', label="Seasonal Naive Forecast, RMSE = " + str(round(rmse, 2)))
plt.title("Seasonal Naive Forecast Vs Original Observation (Apr 2016 - Mar 2017)")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.legend(loc="upper right")
fig.autofmt_xdate(rotation=90)
plt.show(block=False)
fig.savefig("Seasonal Naive Forecast Vs Original Observation.pdf")
###################################################################################################################
##                                   SEASONAL ARIMA FORECAST                                                      #
###################################################################################################################
#print("Seasonal ARIMA Forecast")
#print("-----------------------")
## Create a list for the training data to start at different periods.
#range_grid = (
#slice(0, 4, 1), slice(0, 3, 1), slice(0, 4, 1), slice(0, 4, 1), slice(0, 3, 1), slice(0, 4, 1), slice(12, 13, 1))
#
#for period, train in trains:
#    train_season = ARIMA_series_decompose(period, train_01)
#    ARIMA_ACF_PACF(period, train_season)
#    optimal = brute(object_func, range_grid, args=(train,), finish=None)
#    optimal_converted = [int(x) for x in optimal]
#    predicted = ARIMA_calc_train_test_RMSE(period, optimal_converted)
#    ARIMA_residual_plot(period, test, predicted)
#
#elapsed_time = time.time() - start_time
#print("Time taken to complete : ", elapsed_time, "secs")

###################################################################################################################
##                           Holt Winters Seasonal Exponential Smoothing Forecast (HWSES) FORECAST                #
###################################################################################################################
print("Holt Winters Seasonal Exponential Smoothing Forecast")
print("----------------------------------------------------")

x0 = train_01.index
x1 = np.array([dt.datetime.strftime(d,'%Y-%m') for d in x0])
y1 = train_01[0:]
y1_list = y1.tolist()

# 1.0 Train period 1. Jan 1994 to Mar 2016 and Test period Apr 2016 to Mar 2017
x1full_list = np.array([dt.datetime.strptime(d1f, '%Y-%m') for d1f in months])
y1full = car_sales[0:]
y1full_list = y1full.tolist()

y2_a = np.array(test)

# Test period: Apr 2016 to March 2017 are for forecasting validation
x2 = np.array(test.index.strftime('%Y-%m')) 
y2 = test[0:]
y2_list = y2.tolist()

# 2.1 Train period: Jan 2000 to Mar 2016
x5 = np.array(train_02.index.strftime('%Y-%m'))  
x5_list = np.array([dt.datetime.strptime(d5, '%Y-%m') for d5 in x5])
y5 = train_02[0:]
y5_list = y5.tolist()
y5_a = np.array(y5_list) 

# 2.2 Train period: Jan 2000 to Mar 2016 and Test period Apr 2016 to Mar 2017 
car_sales_012000_032017 = car_sales[(car_sales.index >= '2000-01-01')] 
x5full = np.array(car_sales_012000_032017.index.strftime('%Y-%m')) 
x5full_list = np.array([dt.datetime.strptime(d5f, '%Y-%m') for d5f in x5full]) 
y5full = car_sales_012000_032017[0:]
y5full_list = y5full.tolist()

# 3.1 Train period: Jan 2009 to Mar 2016 (to avoid the outliner with a drop in 2009)
x4 = np.array(train_03.index.strftime('%Y-%m'))  
x4_list = np.array([dt.datetime.strptime(d4, '%Y-%m') for d4 in x4])
y4 = train_03[0:]
y4_list = y4.tolist()

# 3.2 Train period: Jan 2009 to Mar 2016 and Test period Apr 2016 to Mar 2017
car_sales_012009_032017 = car_sales[(car_sales.index >= '2009-01-01')] 
x4full = np.array(car_sales_012009_032017.index.strftime('%Y-%m')) 
x4full_list = np.array([dt.datetime.strptime(d4f, '%Y-%m') for d4f in x4full]) 
y4full = car_sales_012009_032017[0:]
y4full_list = y4full.tolist()

x91 = np.hstack ((x1,x2)) # Jan 1994 to Mar 2017
x92 = np.hstack ((x4,x2)) # Jan 2009 to Mar 2017
x93 = np.hstack ((x5,x2)) # Jan 2000 to Mar 2017

# Start

# Function to caculate the MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ----------------------------------------------------------
# Period 1: Jan 1994 to Mar 2016 with forecast to Mar 2017
# ----------------------------------------------------------

fig = plt.figure()  
fig = plt.figure(figsize=(10,6))
plt.plot(car_sales)
plt.title("Figure HW1. Number of Car sales in Australia from January 1994 to March 2017")
plt.xlabel("Months")
plt.ylabel("Number of Sales")
plt.show(block=False)

fc1 = 12 # 12 months forecasting
h1 = 12 # Number of seasons

y_smoothed_mult1, y_forecast_vals1, alpha1, beta1, gamma1, rmse1 = hw.multiplicative(y1_list,fc=fc1,m=h1)

plt.plot(x1full_list,y1full_list,label = 'Observed')
plt.plot(x1full_list,y_smoothed_mult1[:-1], '-r',color= "red",label ='Multiplicative' + 'rmse = '+str(np.round(rmse1,2)))
plt.title('Figure HW2. Number of Car Sales, Holt-Winters Multiplicative Method, January 1994 -March 2017: alpha = '+str(np.round(alpha1,2)) +' ,beta = '+ str(np.round(beta1,2))+',gamma = '+ str(np.round(gamma1,2)))
plt.xlabel('Year')
plt.legend(loc="lower right")
plt.ylabel('Number of Sales')

fig = plt.figure()
fig = plt.figure(figsize=(10,6))
plt.plot(y_forecast_vals1)
plt.title('Figure HW3. Estimated Seasonal Components')
plt.xlabel("Months")
plt.ylabel("Number of Sales")

forecast1 = np.array(y_forecast_vals1)

print ('Summary on Holt-Winters Multiplicative method with training period Jan 1994 to Mar 2016')
print("RMSE: " + str(np.round(rmse1,2)))
print("MAE: " + str(mae(y2_a, forecast1))) # y2_a refers to Apr 2016 to Mar 2017
print("MAPE: " + str(mean_absolute_percentage_error(y2_a, forecast1)))
print("MSE: " + str(mse(y2_a, forecast1)))

# ----------------------------------------------------------
# Period 2: Jan 2000 to Mar 2016 with forecast to Mar 2017
# ----------------------------------------------------------

fig = plt.figure()  
plt.plot(car_sales_012000_032017)
plt.title("Figure HW4. Number of Car sales in Australia from January 2000 to March 2017")
plt.xlabel("Months")
plt.ylabel("Number of Sales")
plt.show(block=False)

fc5 = 12 # 12 months forecasting
h5 = 12 # Number of seasons

y_smoothed_mult5, y_forecast_vals5, alpha5, beta5, gamma5, rmse5 = hw.multiplicative(y5_list,fc=fc5,m=h5)

plt.plot(x5full_list,y5full_list,label = 'Observed')
plt.plot(x93,y_smoothed_mult5[:-1], '-r',color= "red",label ='Multiplicative' + 'rmse5 = '+str(np.round(rmse5,2)))
plt.title('Figure HW5. Number of Car Sales, Holt-Winters Multiplicative Method, January 2000 -March 2017: alpha = '+str(np.round(alpha5,2)) +' ,beta = '+ str(np.round(beta5,2))+',gamma = '+ str(np.round(gamma5,2)))
plt.xlabel('Year')
plt.legend(loc="lower right")
plt.ylabel('Number of Sales')

forecast5 = np.array(y_forecast_vals5)

print ('Summary on Holt-Winters Multiplicative method with training period Jan 2010 to Mar 2016')
print("RMSE: " + str(np.round(rmse5,2)))
print("MAE: " + str(mae(y2_a, forecast5))) # y2_a refers to Apr 2016 to Mar 2017
print("MAPE: " + str(mean_absolute_percentage_error(y2_a, forecast5)))
print("MSE: " + str(mse(y2_a, forecast5)))

# ----------------------------------------------------------
# Period 3: Jan 2009 to Mar 2016 with forecast to Mar 2017
# ----------------------------------------------------------

fig = plt.figure()  
plt.plot(car_sales_012009_032017)
plt.title("Figure HW6. Number of Car sales in Australia from January 2009 to March 2017")
plt.xlabel("Months")
plt.ylabel("Number of Sales")
plt.show(block=False)

fc4 = 12 # 12 months forecasting
h4 = 12 # Number of seasons

y_smoothed_mult4, y_forecast_vals4, alpha4, beta4, gamma4, rmse4 = hw.multiplicative(y4_list,fc=fc4,m=h4)

plt.plot(x4full_list,y4full_list,label = 'Observed')

plt.plot(x92,y_smoothed_mult4[:-1], '-r',color= "red",label ='Multiplicative' + 'rmse4 = '+str(np.round(rmse4,2)))
plt.title('Figure HW7. Number of Car Sales, Holt-Winters Multiplicative Method, January 2009 -March 2017: alpha = '+str(np.round(alpha4,2)) +' ,beta = '+ str(np.round(beta4,2))+',gamma = '+ str(np.round(gamma4,2)))
plt.xlabel('Year')
plt.legend(loc="lower right")
plt.ylabel('Number of Sales')

forecast4 = np.array(y_forecast_vals4)

print ('Summary on Holt-Winters Multiplicative method with training period Jan 2009 to Mar 2016')
print("RMSE: " + str(np.round(rmse4,2)))
print("MAE: " + str(mae(y2_a, forecast4))) # y2_a refers to Apr 2016 to Mar 2017
print("MAPE: " + str(mean_absolute_percentage_error(y2_a, forecast4)))
print("MSE: " + str(mse(y2_a, forecast4)))

###################################################################################################################
##                                   5. NEURAL NETWORK FORECAST                                                      #
###################################################################################################################
print("Neural Network Forecast")
print("-----------------------")
# Try for 3 period, from 01-Apr-1994, 01-Apr-2000, 01-Apr-2009
nn_data_list = []
nn_data_01 = car_sales[car_sales.index >= '1994-04-01']
nn_data_list.append(('Period - Apr 1994 - Mar 2016', nn_data_01))
nn_data_02 = car_sales[car_sales.index >= '2000-04-01']
nn_data_list.append(('Period - Apr 2000 - Mar 2016', nn_data_02))
nn_data_03 = car_sales[car_sales.index >= '2009-04-01']
nn_data_list.append(('Period - Apr 2009 - Mar 2016', nn_data_03))
time_start = time.time() 
# Calling Neural Network function  
for period, data in nn_data_list:
    generate_LSTM(period, data)
# Check for computation time
elapsed_time = time.time() - start_time
print("Time taken to complete Neural Network : ", elapsed_time, "secs")    

##################################################################################################################
#                                     COMBINED FORECAST                                                      #
##################################################################################################################
print("Combination Forecast")
print("-----------------------")

# ----------------------------------------------------------
# OPTION 1:
# For combined method we will use data from Jan 2010 for :
# 1. SARIMAX
# 2. HOLT WINTERS MULTIPLICATIVE
# ----------------------------------------------------------
    
 #Data value should be from Jan 2010
combined_train = car_sales[(car_sales.index < '2016-04-01') & (car_sales.index >= '2000-01-01')]
combined_test = car_sales[car_sales.index >= '2016-04-01']
combine_data = car_sales[car_sales.index >= '2000-01-01']
combined_train_len = len(combined_train)
combined_test_len = len(combined_test)

# Fit the models
# Model 1 - SARIMAX model
m = 12
sarima_model = smapi.tsa.statespace.SARIMAX(combined_train.values, order=(1,1,1), seasonal_order=(3,1,3,m))
sarima_result = sarima_model.fit(disp=False)
sarima_result.summary()
# Model 2 - Holt Winters Multiplicative
y_smoothed, y_forecast, alpha, beta, gamma, rmse = hw.multiplicative(list(combined_train.values), fc=combined_test_len, m=m)
# Determine optimal weight
residual1 = np.resize(sarima_result.resid, (combined_train_len,))
residual2 = np.resize(combined_train.values - y_smoothed[:combined_train_len], (combined_train_len,))

covariance = np.cov(residual1, residual2)

# variance
var1 = covariance[0][0]
var2 = covariance[1][1]

# correlation coefficient
rho = covariance[0][1] / (np.sqrt(var1*var2))

# Optimise w
wopt1 = (var2 - rho * np.sqrt(var1*var2))/(var1+var2-2*rho*np.sqrt(var1*var2))
# Coefficient is for model 1
wopt2 = 1 - wopt1

# Calculate the forecast
forecast1 = sarima_result.forecast(combined_test_len)
forecast2 = np.array(y_forecast)

# Combination
forecast = wopt1 * forecast1 + wopt2 * forecast2
np.reshape(forecast2, (combined_test_len,))
# Plot the 2 models with combination forecast for comparison
fig = plt.figure(figsize=(10, 8))
plt.plot(combined_test, label= "Test Data")
plt.plot(test.index, forecast, label="Combined Forecasts")
plt.plot(test.index, forecast1, label="Sarimax")
plt.plot(test.index, forecast2, label="Holt Winters Multiplicative")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.title("Combination method using weight for SARIMAX and HW period Apr 2016 to Mar 2017")
plt.legend(loc="upper right")
plt.show(block=False)
fig.savefig("Combination_method_option_1.pdf")

print("Combination method using weight for SARIMAX and HW period Apr 2016 to Mar 2017")
# Use metric to analyze performance
print("Combined MSE: {0:.2f}".format(mse(combined_test.values, forecast)))
print("SARIMAX MSE: {0:.2f}".format(mse(combined_test.values, forecast1)))
print("Holt Winters MSE: {0:.2f}".format(mse(combined_test.values, forecast2)))

# Estimate RMSE
print("Combined RMSE: {0:.2f}".format(np.sqrt(mse(combined_test.values, forecast))))
print("SARIMAX RMSE: {0:.2f}".format(np.sqrt(mse(combined_test.values, forecast1))))
print("Holt Winters RMSE: {0:.2f}".format(np.sqrt(mse(combined_test.values, forecast2))))

# Estimate MAPE
print("Combined MAPE: {0:.2f}".format(mean_absolute_percentage_error(combined_test.values, forecast)))
print("SARIMAX MAPE: {0:.2f}".format(mean_absolute_percentage_error(combined_test.values, forecast1)))
print("Holt Winters MAPE: {0:.2f}".format(mean_absolute_percentage_error(combined_test.values, forecast2)))

# ----------------------------------------------------------
# OPTION 2:
# For combined method we will use data from Jan 2010 for :
# 1. SARIMAX
# 2. HOLT WINTERS MULTIPLICATIVE
# 3. SEASONAL NAIVE FORECAST
# WE ALSO WILL USE AVERAGE METHOD INSTEAD OF WEIGHT METHOD
# ----------------------------------------------------------
# Seasonal Naive Bayes
actual_value = [x for x in combined_train]
predicted_value = []
for i in range(len(combined_test)):
    predicted_value.append(actual_value[-12])
    actual_value.append(combined_test[i])
forecast3 = np.asarray(predicted_value)
# use naive combination method (average)
forecast = (forecast1 + forecast2 + forecast3)/3
# Plot the three models and combined forescast
fig = plt.figure(figsize=(10, 8))
plt.plot(combined_test, label= "Test Data")
plt.plot(test.index, forecast, label="Combined Forecasts")
plt.plot(test.index, forecast1, label="Sarimax")
plt.plot(test.index, forecast2, label="Holt Winters Multiplicative")
plt.plot(test.index, forecast2, label="Seasonal Naive Forecast")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.title("Combination method using average for SARIMAX, HW and Seasonal Naive period Apr 2016 to Mar 2017")
plt.legend(loc="upper right")
plt.show(block=False)
fig.savefig("Combination_method_option_2.pdf")   

# Use metric to analyze performance
print("Combination method using average for SARIMAX, HW and Seasonal Naive period Apr 2016 to Mar 2017")
print("Combined MSE: {0:.2f}".format(mse(combined_test.values, forecast)))
print("SARIMAX MSE: {0:.2f}".format(mse(combined_test.values, forecast1)))
print("Holt Winters MSE: {0:.2f}".format(mse(combined_test.values, forecast2)))
print("Seasonal Naive MSE: {0:.2f}".format(mse(combined_test.values, forecast3)))
# Estimate RMSE
print("Combined RMSE: {0:.2f}".format(np.sqrt(mse(combined_test.values, forecast))))
print("SARIMAX RMSE: {0:.2f}".format(np.sqrt(mse(combined_test.values, forecast1))))
print("Holt Winters RMSE: {0:.2f}".format(np.sqrt(mse(combined_test.values, forecast2))))
print("Seasonal Naive RMSE: {0:.2f}".format(np.sqrt(mse(combined_test.values, forecast3))))
# Estimate MAPE
print("Combined MAPE: {0:.2f}".format(mean_absolute_percentage_error(combined_test.values, forecast)))
print("SARIMAX MAPE: {0:.2f}".format(mean_absolute_percentage_error(combined_test.values, forecast1)))
print("Holt Winters MAPE: {0:.2f}".format(mean_absolute_percentage_error(combined_test.values, forecast2)))
print("Seasonal Naive MAPE: {0:.2f}".format(mean_absolute_percentage_error(combined_test.values, forecast3)))

