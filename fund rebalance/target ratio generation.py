# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:47:18 2020

@author: bb19x
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize


raw_data=pd.read_excel(r"C:\Users\bb19x\OneDrive\桌面\何もない\Bellman Euqtion new SAA\csv_data\1995_2019 daily data.xlsx").iloc[611:]
#10 : 611  5 : 1871
daily_data=pd.DataFrame()

daily_data["SPX_Index"]=raw_data["SPX Index"]
daily_data["SXXP_Index"]=raw_data["SXXP Index"]
daily_data["NKY_Index"]=raw_data["NKY Index"]
daily_data["SHCOMP_Index"]=raw_data["SHCOMP Index"]
daily_data["HSCEI_Index"]=raw_data["HSCEI Index"]
daily_data["TWSE_Index"]=raw_data["TWSE Index"]
daily_data["KOSPI_Index"]=raw_data["KOSPI Index"]
daily_data["SENSEX_Index"]=raw_data["SENSEX Index"]
daily_data["MXSO_Index"]=raw_data["MXSO Index"]
daily_data["MXMU_Index"]=raw_data["MXMU Index"]
daily_data["MXLA_Index"]=raw_data["MXLA Index"]

alpha=4
saa_number=11
length_of_one_period=21
length_of_time=len(daily_data)
length_of_train=120
N=11 #11項SA

#%%

#用來計算初始投組目標比率(無考慮5%門檻)
def pre_target_portfolio(pre_target_return,pre_target_cov):
    
    def Utility(weight):
        return -np.dot(weight,pre_target_return)+0.5*alpha*np.dot(weight.T,np.dot(pre_target_cov,weight))

    def constraint1(weight):
        return 1-sum(weight)

    weight0=np.full(saa_number,1/saa_number)
    b=(0,1)
    bnds=(b,)*saa_number
    con1={"type":"eq","fun":constraint1}
    cons=[con1]
    sol = minimize( Utility, weight0, method="SLSQP", bounds=bnds, constraints=cons, tol=1e-18)  
    
    return sol.x

#用來計算篩選過後的投組的目標比率(有考慮5%門檻)
def target_portfolio(period_return,period_cov,index1):
    
    dim=len(index1)
    
    def Utility(weight):
        return -np.dot(weight,period_return)+0.5*alpha*np.dot(weight.T,np.dot(period_cov,weight))

    def constraint1(weight):
        return 1-sum(weight)

    weight0=np.full(dim,1/dim)
    b=(0,1)
    bnds=(b,)*dim
    con1={"type":"eq","fun":constraint1}
    cons=[con1]
    sol = minimize( Utility, weight0, method="SLSQP", bounds=bnds, constraints=cons, tol=1e-18) 
        
    return sol.x

#%%
k=0.05
period_data=daily_data[:length_of_train*length_of_one_period]
train_data=period_data.copy()
daily_return_train_data=train_data/train_data.shift(1)-1
pre_target_return=daily_return_train_data.mean().values*length_of_one_period
pre_target_cov=daily_return_train_data.cov().values*length_of_one_period
pre_target_ratio=pre_target_portfolio(pre_target_return,pre_target_cov)
index1=np.where(pre_target_ratio>=k)[0]

train_data=train_data[train_data.columns[index1]]
daily_return=train_data/train_data.shift(1)-1
period_return=daily_return.mean().values*length_of_one_period
period_cov=daily_return.cov().values*length_of_one_period
target_ratio=target_portfolio(period_return,period_cov,index1) # save the ratio of certain target assets

while len(np.where(target_ratio<k)[0])>0:
    index1=np.delete(index1,np.where(target_ratio<k)[0])
    train_data=period_data[period_data.columns[index1]]
    daily_return=train_data/train_data.shift(1)-1
    period_return=daily_return.mean().values*length_of_one_period
    period_cov=daily_return.cov().values*length_of_one_period
    target_ratio=target_portfolio(period_return,period_cov,index1)

saa_target_ratio=np.zeros(daily_data.shape[1])
for i in range(len(index1)):
    saa_target_ratio[index1[i]]=target_ratio[i]

period_target_ratio=np.copy(saa_target_ratio)

#此迴圈每3期變動1次目標比率才需要
for i in range(2):
      period_target_ratio=np.vstack((period_target_ratio,saa_target_ratio))

#每3期變動1次目標比率 : range(1,int(np.floor((length_of_time-length_of_train*length_of_one_period)/length_of_one_period/3)))
#每1期變動1次目標比率 : range(1,int(np.floor((length_of_time-length_of_train*length_of_one_period)/length_of_one_period)))

for period in range(1,int(np.floor((length_of_time-length_of_train*length_of_one_period)/length_of_one_period/3))):
    period_data=daily_data[(3*period)*length_of_one_period:(length_of_train+3*period)*length_of_one_period]
    train_data=period_data.copy()
    
    daily_return_train_data=train_data/train_data.shift(1)-1
    pre_target_return=daily_return_train_data.mean().values*length_of_one_period
    pre_target_cov=daily_return_train_data.cov().values*length_of_one_period
    pre_target_ratio=pre_target_portfolio(pre_target_return,pre_target_cov)
    
    index1=np.where(pre_target_ratio>=k)[0]
    train_data=train_data[train_data.columns[index1]]
    daily_return=train_data/train_data.shift(1)-1
    period_return=daily_return.mean().values*length_of_one_period
    period_cov=daily_return.cov().values*length_of_one_period
    period_std=daily_return.var().values*np.sqrt(length_of_one_period)
    target_ratio=target_portfolio(period_return,period_cov,index1)
    
    while len(np.where(target_ratio<k)[0])>0:
        index1=np.delete(index1,np.where(target_ratio<k)[0])
        train_data=period_data[period_data.columns[index1]]
        daily_return=train_data/train_data.shift(1)-1
        period_return=daily_return.mean().values*length_of_one_period
        period_cov=daily_return.cov().values*length_of_one_period
        period_std=daily_return.std().values
        target_ratio=target_portfolio(period_return,period_cov,index1)
    
    saa_target_ratio=np.zeros(daily_data.shape[1])
    for i in range(len(index1)):
        saa_target_ratio[index1[i]]=target_ratio[i]
    
    period_target_ratio=np.vstack((period_target_ratio,saa_target_ratio))
    
    #此迴圈每3期變動1次目標比率才需要
    for i in range(2):
        period_target_ratio=np.vstack((period_target_ratio,saa_target_ratio))
 
#%%

#這一塊程式碼我是用來看SAA目標比率變動幅度(有5%門檻VS沒有5%門檻)

diff_pre=[]
for i in range(1,len(period_target_ratio)):
    diff_pre.append(period_target_ratio[i]-period_target_ratio[i-1]) 
    
diff_5=[]
for i in range(1,len(period_target_ratio)):
    diff_5.append(period_target_ratio[i]-period_target_ratio[i-1])

abs_pre=np.sum(np.abs(diff_pre))
abs_5=np.sum(np.abs(diff_5))

print(abs_pre)
print(abs_5)
#%%
os.chdir(r"C:\Users\bb19x\OneDrive\桌面\何もない\dynamic ratio test\fund rebalance")

with open('period target ratio (change every period).txt', 'w') as outfile:
    json.dump(period_target_ratio.tolist(), outfile)

# with open('period target ratio.txt', 'w') as outfile:
#    json.dump(period_target_ratio.tolist(), outfile)        