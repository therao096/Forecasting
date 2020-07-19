# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:37:33 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime,time

cola=pd.read_excel("F:\\EXCEL R\\ASSIGNMENTS\\forecasting\\CocaCola_Sales_Rawdata.xlsx")

cola.Sales.plot()
###cola['date']=pd.to_datetime(cola.Quarter,format="%b=%y")

heatmap_y_month = pd.pivot_table(data=cola,values="Sales",index="Quarter",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")
sns.boxplot(x="Quarter",y="Sales",data=cola)
sns.lineplot(x="Quarter",y="Sales",data=cola)


cola.Sales.plot(label="org")
for i in range(2,24,6):
    cola["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

Train = cola.head(30)
Test = cola.tail(12)    


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

##simple exponential
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) ##16.64


###holt method
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales)
##8.997


###holt winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)
###4.54


#### Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)
####7.59

###out of the four models, holt winter exponential smoothing with additive seasonality and additive trend has least MAPE 4.54

pred_test = pd.Series(hwe_model_add_add.predict(start = cola.index[0],end = cola.index[-1]))

pred_test.index = cola.index
MAPE(pred_test,cola.Sales)
##5.131

##visualization
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")