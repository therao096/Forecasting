# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:07:27 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime,time

air=pd.read_excel("F:\\EXCEL R\\ASSIGNMENTS\\forecasting\\Airlines+Data.xlsx")

air.Passengers.plot() ##timeseries plot
air["Date"] = pd.to_datetime(air.Month,format="%b-%y")

air["month"] = air.Date.dt.strftime("%b")
air["year"] = air.Date.dt.strftime("%Y")

##visualize a heatmap

heatmap_y_month = pd.pivot_table(data=air,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")


##boxplts
sns.boxplot(x="month",y="Passengers",data=air)
sns.boxplot(x="year",y="Passengers",data=air)
sns.lineplot(x="year",y="Passengers",data=air)
air.Passengers.plot(label="org")

###for moving average
for i in range(2,24,6):
    air["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)




##timeseries decomposition plot
decompose_ts_add = seasonal_decompose(air.Passengers,model="additive",freq=12)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(air.Passengers,model="multiplicative",freq=12)
decompose_ts_mul.plot()

##acf and pacf plot
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(air.Passengers,lags=12)
tsa_plots.plot_pacf(air.Passengers,lags=12)


##split data
Train=air.head(84)
Test=air.tail(12)

####function for Mape value

def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

##simple exponential method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers) ##14.2354

###holt method
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers) ##11.84

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers) ##1.61

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers) ##2.82


##hwe_add_add has least MAPE value
pred_test = pd.Series(hwe_model_add_add.predict(start = air.index[0],end = air.index[-1]))

pred_test.index = air.index
MAPE(pred_test,air.Passengers)
##3.513

plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
