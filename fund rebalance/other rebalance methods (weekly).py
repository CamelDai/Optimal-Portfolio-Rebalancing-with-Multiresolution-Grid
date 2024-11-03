import numpy as np
import pandas as pd
import json
# import os

saa_raw_data=pd.read_excel(r"C:\Users\bb19x\OneDrive\桌面\何もない\Bellman Euqtion new SAA\csv_data\1995_2019 daily data.xlsx").iloc[611:]
fund_daily_data=pd.read_excel(r"C:\Users\bb19x\OneDrive\桌面\何もない\unbundle\partial daily price of funds.xlsx").drop(["ISIN 代碼"],axis=1).fillna(0).values
unbundle_data=pd.read_excel(r"C:\Users\bb19x\OneDrive\桌面\何もない\unbundle\better unbundle (partial processed).xlsx")

with open('period_fund_target_ratio.txt') as json_file:
    period_fund_target_ratio = json.load(json_file)

period_fund_target_ratio=np.array(period_fund_target_ratio)

with open('period_alpha.txt') as json_file:
    period_alpha = json.load(json_file)

with open('period target ratio.txt') as json_file:
    saa_target = json.load(json_file)

saa_target=np.array(saa_target)
price_data=pd.DataFrame()

price_data["SPX_Index"]=saa_raw_data["SPX Index"]
price_data["SXXP_Index"]=saa_raw_data["SXXP Index"]
price_data["NKY_Index"]=saa_raw_data["NKY Index"]
price_data["SHCOMP_Index"]=saa_raw_data["SHCOMP Index"]
price_data["HSCEI_Index"]=saa_raw_data["HSCEI Index"]
price_data["TWSE_Index"]=saa_raw_data["TWSE Index"]
price_data["KOSPI_Index"]=saa_raw_data["KOSPI Index"]
price_data["SENSEX_Index"]=saa_raw_data["SENSEX Index"]
price_data["MXSO_Index"]=saa_raw_data["MXSO Index"]
price_data["MXMU_Index"]=saa_raw_data["MXMU Index"]
price_data["MXLA_Index"]=saa_raw_data["MXLA Index"]

tickers=[]

for i in price_data:
    if i=="Dates":
        pass
    else:
        tickers.append(i)
    
daily_returns = pd.DataFrame()

for t in tickers:
    daily_returns[t] = price_data[t]/price_data[t].shift(1)-1

saa_number=11
length_of_one_period=21
length_of_sub_period=7 #sub_period是7個交易日。雖然7個交易日超過一個禮拜，但為了方便我將檔名用weekly做標記
length_of_time=len(price_data)
length_of_train=120

C_buy=0.015
C_sell=0
num_saa=price_data.shape[1]
Fund=np.array(unbundle_data.iloc[:,1:]).T
N=Fund.shape[1]
num_saa=11

#計算基金期望報酬與共變異數矩陣
def funds_returns_cov(mu_period,cov_period):
    
    def fund_cov(i,j):
        covariance=0
        for p in range(num_saa):
            for q in range(num_saa):
                covariance=covariance+Fund.T[i][p]*Fund.T[j][q]*cov_period[p][q]
        return covariance
    
    Fund_mu=np.zeros(N)
    for i in range(N):
        Fund_mu[i]=np.dot(Fund.T[i],mu_period)
        
    Fund_cov=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Fund_cov[i][j]=fund_cov(i,j)
    
    return Fund_mu,Fund_cov

#計算效用
def utility(weight,period_return,period_cov,alpha):
    
    return np.dot(weight,period_return)-0.5*alpha*np.dot(weight.T,np.dot(period_cov,weight))

#計算交易成本
def TC(w1,w2):
    C=[]
    for k in range(N):
        if w2[k]-w1[k]>0:
            C.append(C_buy)
        else:
            C.append(C_sell)
    return np.dot(np.array(C),abs(w2-w1))
    


#%% periodic rebalance

tracking_error=[]
trading_cost=[]
periodic=36
weights=period_fund_target_ratio[0]
#period_fund_target_ratio.shape[0]

for sub_period in range(int(len(saa_target)*3)):
    
    print("%d th time"%sub_period)
    if sub_period % 9 == 0:
        alpha=period_alpha[int(sub_period/3)]
        target=period_fund_target_ratio[int(sub_period/3)]
        saa_period_daily_returns=daily_returns[sub_period*length_of_sub_period:(sub_period+length_of_train*3)*length_of_sub_period]
        saa_period_return=saa_period_daily_returns.mean().values*length_of_sub_period
        saa_period_cov=saa_period_daily_returns.cov().values*length_of_sub_period
    
        fund_return,fund_cov=funds_returns_cov(saa_period_return,saa_period_cov)
    
    after_sub_period_data=fund_daily_data[length_of_sub_period*sub_period:length_of_sub_period*(sub_period+1)]
    after_sub_period_return=np.nan_to_num(after_sub_period_data[-1]/after_sub_period_data[0])
    
    portfolio_growth=np.dot(weights,after_sub_period_return)
    weight_after_sub_period_temp=weights*after_sub_period_return/portfolio_growth
    
    if sub_period % periodic==0:
        
        weights=np.copy(target)
        tracking_error.append(0)
        trading_cost.append(TC(weight_after_sub_period_temp,target))
       
    else:
        
        weights=np.copy(weight_after_sub_period_temp)
        tracking_error.append(utility(target,fund_return,fund_cov,alpha)-utility(weights,fund_return,fund_cov,alpha))
        trading_cost.append(0)

print("Total tracking error is %f"%(sum(tracking_error)))
print("Total trading cost is %f"%(sum(trading_cost)))
print("Total cost is %f"%(sum(tracking_error)+sum(trading_cost)))
       
#%% tolerance rebalance

tracking_error=[]
trading_cost=[]
weights=period_fund_target_ratio[0]
tol=0.26

for period in range(int(len(saa_target)/3)):
    
    print("%d th time"%period)
    saa_period_daily_returns=daily_returns[3*period*length_of_one_period:(3*period+length_of_train)*length_of_one_period]
    saa_period_return=saa_period_daily_returns.mean().values*length_of_sub_period
    saa_period_cov=saa_period_daily_returns.cov().values*length_of_sub_period
    
    fund_return,fund_cov=funds_returns_cov(saa_period_return,saa_period_cov)
    
    after_3_peroid_data=fund_daily_data[length_of_one_period*period*3:length_of_one_period*(period+1)*3]
    
    for first_3_periods in range(9):
        after_one_period_data=after_3_peroid_data[first_3_periods*length_of_sub_period:(1+first_3_periods)*length_of_sub_period]
        after_one_period_return=np.nan_to_num(after_one_period_data[-1]/after_one_period_data[0])
    
        portfolio_growth=np.dot(weights,after_one_period_return)
        weight_after_one_period_temp=weights*after_one_period_return/portfolio_growth
        alpha=period_alpha[period*3]
    
        if np.sum(abs(weight_after_one_period_temp-period_fund_target_ratio[period])>=tol)>0:
        
            weights=np.copy(period_fund_target_ratio[period*3])
            tracking_error.append(0)
            trading_cost.append(TC(weight_after_one_period_temp,period_fund_target_ratio[period*3]))
        
        else:
        
            weights=np.copy(weight_after_one_period_temp)
            tracking_error.append(utility(period_fund_target_ratio[period*3],fund_return,fund_cov,alpha)-utility(weights,fund_return,fund_cov,alpha))
            trading_cost.append(0)

print("Total tracking error is %f"%(sum(tracking_error)))
print("Total trading cost is %f"%(sum(trading_cost)))
print("Total cost is %f"%(sum(tracking_error)+sum(trading_cost)))
print(len(np.nonzero(trading_cost)[0]))
#%% buy and hold
        
tracking_error=[]
weights=period_fund_target_ratio[0]

for period in range(int(len(saa_target)/3)):
    
    print("%d th time"%period)
    saa_period_daily_returns=daily_returns[3*period*length_of_one_period:(3*period+length_of_train)*length_of_one_period]
    saa_period_return=saa_period_daily_returns.mean().values*length_of_sub_period
    saa_period_cov=saa_period_daily_returns.cov().values*length_of_sub_period
    
    fund_return,fund_cov=funds_returns_cov(saa_period_return,saa_period_cov)
    
    after_3_peroid_data=fund_daily_data[length_of_one_period*period*3:length_of_one_period*(period+1)*3]
    
    for first_3_periods in range(9):
        after_one_period_data=after_3_peroid_data[first_3_periods*length_of_sub_period:(1+first_3_periods)*length_of_sub_period]
        after_one_period_return=after_one_period_data[-1]/after_one_period_data[0]
    
        portfolio_growth=np.dot(weights,after_one_period_return)
        weight_after_one_period_temp=weights*after_one_period_return/portfolio_growth
        alpha=period_alpha[period*3]
    
        tracking_error.append(utility(period_fund_target_ratio[period],fund_return,fund_cov,alpha)-utility(weights,fund_return,fund_cov,alpha))

print("Total tracking error is %f"%(sum(tracking_error)))
print("Total trading cost is %f"%(0))
print("Total cost is %f"%(sum(tracking_error)))
    
    
       

