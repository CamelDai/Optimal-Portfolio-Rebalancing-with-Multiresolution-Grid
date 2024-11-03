# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:18:17 2020

@author: bb19x
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import os
from itertools import product
import time

os.chdir(r"C:\Users\bb19x\OneDrive\桌面\何もない\dynamic ratio test\fund rebalance")

# unbundle 完成以後的基金
unbundle_data=pd.read_excel(r"C:\Users\bb19x\OneDrive\桌面\何もない\unbundle\better unbundle (partial processed).xlsx")

# SAA 每日價格
saa_raw_data=pd.read_excel(r"C:\Users\bb19x\OneDrive\桌面\何もない\Bellman Euqtion new SAA\csv_data\1995_2019 daily data.xlsx").iloc[611:]

# 基金每日價格
fund_daily_data=pd.read_excel(r"C:\Users\bb19x\OneDrive\桌面\何もない\unbundle\partial daily price of funds.xlsx").drop(["ISIN 代碼"],axis=1).values
#從2007/1/1 開始 #SAA目標比例設定估計期間從1997/6/11開始

# 提取每期 SAA 目標比率
with open('period target ratio.txt') as json_file:
    saa_target = json.load(json_file)

saa_target=np.array(saa_target)

# 以下計算每項 SA 的每日報酬

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

num_saa=11 # SA 數量

#%%
# EXCEL檔裡面的基金權重是橫的，這裡將權重以直行來表示
Fund=np.array(unbundle_data.iloc[:,1:]).T
# 基金總數量
N=Fund.shape[1]

#用來選取特徵性較強的基金逼近目標比率
def select_proper_funds(saa_target):
    
    def error(weight):
        approximate_saa_weight=np.dot(Fund,weight)
        e=(approximate_saa_weight-saa_target)**2
        return sum(e)

    def constraint1(weight):
        return float(1)-sum(weight)

    weight0=np.full(N,1/N)
    b=(0,1)
    bnds=(b,)*N
    con1={"type":"eq","fun":constraint1}
    cons=[con1]

    sol=minimize(error,weight0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18)
    fund_index=np.where(sol.x>0.00001)[0]

    return fund_index

#計算篩選出的基金的期望報酬與彼此之間的共變異數矩陣以及相關係數矩陣
def funds_returns_cor_cov(fund_index,mu_period,cov_period):
    
    fund_portfolio=Fund[:,fund_index]
    num_fund=len(fund_index)
    
    def fund_cov(i,j):
        covariance=0
        for p in range(num_saa):
            for q in range(num_saa):
                covariance=covariance+fund_portfolio.T[i][p]*fund_portfolio.T[j][q]*cov_period[p][q]
        return covariance
    
    Fund_mu=np.zeros(num_fund)
    for i in range(num_fund):
        Fund_mu[i]=np.dot(fund_portfolio.T[i],mu_period)
        
    Fund_cov=np.zeros((num_fund,num_fund))
    for i in range(num_fund):
        for j in range(num_fund):
            Fund_cov[i][j]=fund_cov(i,j)

    Fund_cor=np.zeros((num_fund,num_fund))
    for i in range(num_fund):
        for j in range(num_fund):
            Fund_cor[i][j]=Fund_cov[i][j]/(Fund_cov[i][i]**0.5*Fund_cov[j][j]**0.5)
    
    return Fund_mu,Fund_cov,Fund_cor

#選取最合適的alpha，使得在這alpha下，其最適基金目標比率所轉換成的SAA比率最靠近SAA目標比率
def alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target):
    
    fund_portfolio=Fund[:,fund_index]
    num_fund=len(fund_index)
    alpha_list=np.linspace(1.5,5,351)
    error_test=[]
    
    def Utility(weight):
        return -np.dot(weight,Fund_mu)+0.5*alpha_list[i]*np.dot(weight.T,np.dot(Fund_cov,weight))

    def constraint1(weight):
        return 1-sum(weight)
    
    weight0=np.full(num_fund,1/num_fund)
    b=(0,1)
    bnds=(b,)*num_fund
    con1={"type":"eq","fun":constraint1}
    cons=[con1]
    
    for i in range(len(alpha_list)):
        sol=minimize(Utility,weight0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18)
        error_test.append(sum((np.dot(fund_portfolio,sol.x)-saa_target)**2))
    
    return alpha_list[np.argmin(error_test)]

#求出在最適alpha下的基金最適目標比率
def target_ratio_of_funds_portfolio(alpha,Fund_mu,Fund_cov):
    
    num_fund=len(Fund_mu)
    
    def Utility(weight):
        return -np.dot(weight,Fund_mu)+0.5*alpha*np.dot(weight.T,np.dot(Fund_cov,weight))

    def constraint1(weight):
        return 1-sum(weight)
    
    b=(0,1)
    bnds=(b,)*num_fund
    con1={"type":"eq","fun":constraint1}
    cons=[con1]

    weight0=np.full(num_fund,1/num_fund)
    sol=minimize(Utility,weight0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18)
    
    return sol.x

#可有可無，只是用來確認算出來的基金目標比率效用是否最大
def check_the_real_target(fund_target_ratio,Fund_union_mu,Fund_union_cov,alpha):
    
    def Utility(weight):
        return np.dot(weight,Fund_union_mu)-0.5*alpha*np.dot(weight.T,np.dot(Fund_union_cov,weight))
    
    precision=10**-3
    temp_expand,test=[],[]
    for i in range(len(fund_target_ratio)):
        temp_expand.append([])
    for i in range(len(fund_target_ratio)):
        temp_expand[i].append(fund_target_ratio[i])    
    for i in range(len(fund_target_ratio)):
        if temp_expand[i][-1]<=precision:
            pass
            for j in range(15):
                temp_expand[i].append(temp_expand[i][-1]+precision)
                temp_expand[i].insert(0,temp_expand[i][0]-precision)
    
    temp_states=np.array(list(product(*temp_expand)))
    temp_states=temp_states[abs(np.sum(temp_states,1)-1)<precision*0.1]
    
    for i in range(len(temp_states)):
        test.append(Utility(temp_states[i]))
    
    return temp_states[np.argmax(test)]
    
#每一期因為可能會買進或賣出基金，所以每一期投組基金數量可能會不太一樣，也就是每一期權重的維度大小不一樣
#所以透過這個function將上一期的權重維度和這一期的權重維度相容在一起
def weight_expand(union,index0,index1,weight):
    
    weight=weight.tolist()
    to_be_bought_index=np.sort(list(set(union)-set(index0)))
    to_be_sold_index=np.sort(list(set(union)-set(index1)))
    for i in to_be_bought_index:
        weight.insert(union.index(i),0)
    
    return weight,to_be_bought_index,to_be_sold_index
    
#計算出每個基金權重的上下界以及個基金較有可能的權重
def expand(target_expand,union,to_be_bought_index,to_be_sold_index,Fund_union_mu,Fund_union_std,fund_target_ratio,weight):
    
    k=3
    inter=min(Fund_union_std)*2/3  # 權重之間的間距
    be_bought,be_sold,all_up,all_down,up,down,interval=[],[],[],[],[],[],[]
    length_index=len(union)
    
    for i in to_be_sold_index:
        be_sold.append(union.index(i))
    for i in to_be_bought_index:
        be_bought.append(union.index(i))
    
    for i in range(length_index):
        all_up.append(1+Fund_union_mu[i]+k*Fund_union_std[i])
        all_down.append(1+Fund_union_mu[i]-k*Fund_union_std[i])
    for i in range(length_index):
        up.append(fund_target_ratio[i]*all_up[i]/(np.dot(fund_target_ratio,all_down)+2*k*fund_target_ratio[i]*Fund_union_std[i]))
        down.append(fund_target_ratio[i]*all_down[i]/(np.dot(fund_target_ratio,all_up)-2*k*fund_target_ratio[i]*Fund_union_std[i]))
    for i in range(length_index):
        interval.append([])
    for i in range(length_index):
        interval[i].append(down[i])
        interval[i].append(up[i])   
    
    for i in range(length_index):
        target_expand[i].append(fund_target_ratio[i])
    
    for i in range(length_index):
        if i in be_bought:
            while target_expand[i][-1]<interval[i][1]:
                target_expand[i].append(target_expand[i][-1]+inter)
            while target_expand[i][0]-inter>0:
                target_expand[i].insert(0,target_expand[i][0]-inter)
            target_expand[i].insert(0,0)
        elif i in be_sold:
            while target_expand[i][-1]+inter<interval[i][1] or target_expand[i][-1]+inter < weight[i]:
                target_expand[i].append(target_expand[i][-1]+inter)
        else:
            if fund_target_ratio[i]>weight[i]:
                while target_expand[i][-1]+inter<interval[i][1]:
                    target_expand[i].append(target_expand[i][-1]+inter)
                while (target_expand[i][0]-inter>interval[i][0] or target_expand[i][0]>weight[i]) and target_expand[i][0]-inter>0:
                    target_expand[i].insert(0,target_expand[i][0]-inter)
            else:
                if fund_target_ratio[i]<weight[i]:
                    while target_expand[i][-1]+inter<interval[i][1] or target_expand[i][-1]<weight[i]:
                        target_expand[i].append(target_expand[i][-1]+inter)
                    while target_expand[i][0]-inter>interval[i][0] and target_expand[i][0]-inter>0:
                        target_expand[i].insert(0,target_expand[i][0]-inter)
    
    return target_expand

#建構所有狀態   
def states_construction(target_expand):
    states=np.array(list(product(*target_expand)))
    states=states[abs(np.sum(states,1)-1)<0.0001]
    return states

#檢查目標比率效用是否最大化(可有可無)
def check_the_utility(states,union_return,union_cov,alpha):
    
    def Rce(j):
        return np.dot(union_return,states[j])-0.5*alpha*np.dot(states[j].T,np.dot(union_cov,states[j]))
    
    test=[]
    for i in range(len(states)):
        test.append(Rce(i))
    
    return np.argmax(test)

#狀態建構完以後計算狀態機率轉換矩陣
def transition(states,union_return,union_std,cor):
    
    N,number_of_states=states.shape[1],len(states)
    states_after=np.zeros((number_of_states,number_of_states))
    simulation=20000
    np.random.seed(0)
    rand=np.random.normal(0,1,(simulation,N))
    A,B=np.zeros((simulation,N)),np.linalg.cholesky(cor)
    for i in range(simulation):
        A[i]=np.dot(B,rand[i])
    states_temp=np.zeros((simulation*number_of_states,N))   
    growth=np.zeros((simulation,N))
    for i in range(simulation):
        growth[i]=np.exp((union_return-0.5*union_std**2)+union_std*A[i])
    for i in range(number_of_states):
        for j in range(simulation):
            states_temp[simulation*i+j]=states[i]*growth[j]
    states_temp_weight=np.zeros((len(states_temp),N))
    
    for i in range(len(states_temp)):
        sum_of_weight=np.sum(states_temp[i])
        states_temp_weight[i]=states_temp[i]/sum_of_weight
        
    weight_after=np.zeros((number_of_states,simulation,N))
    for i in range(len(np.zeros((number_of_states,simulation,N)))):
        weight_after[i]=states_temp_weight[i*simulation:(1+i)*simulation]
    
    for i in range(number_of_states):
        for j in range(simulation):
            dis=np.zeros(number_of_states)
            dis=np.sum((np.tile(weight_after[i][j],(number_of_states,1))-states)**2,axis=1)
            states_after[i][np.argmin(dis)]+=1
    Transition=np.zeros((number_of_states,number_of_states))
    for i in range(number_of_states):
        summation=np.sum(states_after[i])
        for j in range(number_of_states):
            Transition[i][j]=states_after[i][j]/summation
    
    return Transition

#執行policy iteration     
def dynamic_programming(states,transition_matrix,union_return,union_cov,final_target_ratio_index,alpha):
    
    beta=np.exp(-0.02/12)
    ite=15
    number_of_states=len(states)
    policy_PI=np.zeros(number_of_states)
    J=np.zeros(number_of_states)
    
    def Rce(j):
        return np.dot(union_return,states[j])-0.5*alpha*np.dot(states[j].T,np.dot(union_cov,states[j]))
    
    def trading_cost(i,j):
        dim=len(union_return)
        C=np.zeros(dim)
        for k in range(dim):
            if states[j][k]-states[i][k]>0:
                C[k]=0.015 # buy
            else:
                C[k]=0 # sell
        return np.dot(C,abs(states[i]-states[j]))
    
    Rce_target=Rce(final_target_ratio_index)
    def tracking_error(j):
        return Rce_target-Rce(j)
    
    first_policy=np.zeros(number_of_states)
    for i in range(number_of_states):
        first_policy[i]=final_target_ratio_index
    
    policy_PI=np.vstack((policy_PI,first_policy))
    
    def policy_evaluation(k):
        J=np.zeros((ite+1,number_of_states))
        for i in range(1,ite):
            for j in range(number_of_states):
                J[i][j]=trading_cost(j,k[j])+tracking_error(k[j])+beta*np.dot(transition_matrix[k[j]],J[i-1])
        return J[-1]

    def policy_improvement(i,J):
        compare=np.zeros(number_of_states)
        for k in range(number_of_states):
            compare[k]= trading_cost(i,k)+tracking_error(k)+beta*np.dot(transition_matrix[k],J)
        policy_PI[-1][i]=np.argmin(compare)
        return min(compare)
    
    while ~(policy_PI[-1]==policy_PI[-2]).all() :
        policy_PI=np.vstack((policy_PI,np.zeros(number_of_states)))
        J=policy_evaluation(policy_PI[-2].astype(int))
        for i in range(number_of_states):
            policy_improvement(i,J)
    
    return policy_PI[-1]

#將因市場波動所造成的權重歸類到我們所建構的狀態之中
def check_weight(weight,states):
    dis=[]
    number_of_assets=len(states)
    for i in range(number_of_assets):
        dis.append(sum((weight-states[i])**2))
    weight_index=np.argmin(dis)
    return weight_index

#歸類到某個狀態以後，查看該狀態的最佳策略
def policy_check(weight,weight_index,policy,states,union_return,union_cov,final_target_ratio_index):
    
    weight=np.array(weight)
    
    def utility(w):
        return np.dot(union_return,w)-0.5*alpha*np.dot(w.T,np.dot(union_cov,w))

    def trading_cost(w1,w2):
        dim=len(union_return)
        C=np.zeros(dim)
        for k in range(dim):
            if w2[k]-w1[k]>0:
                C[k]=0.015 # buy
            else:
                C[k]=0 # sell
        return np.dot(C,abs(w1-w2))

    Rce_target=utility(states[final_target_ratio_index])
    def tracking_error(w):
        return Rce_target-utility(w)
    
    if weight_index!=policy[weight_index]:
        tempt=np.copy(weight)
        weight=np.copy(states[int(policy[weight_index])])
        tracking_error=tracking_error(states[int(policy[weight_index])])
        trading_cost=trading_cost(tempt,states[int(policy[weight_index])])
    else:
        tracking_error=tracking_error(weight)
        trading_cost=0
    
    return trading_cost,tracking_error,weight

#%%
running_time=[]
number_of_states=[]
period_weight=[[]]
Fund=np.array(unbundle_data.iloc[:,1:]).T
length_of_one_period=21
length_of_sub_period=7
length_of_train=120
trading_cost_list,tracking_error_list=[],[] # 用來儲存每一期的交易成本與追蹤誤差
period_alpha=[] # 用來儲存每一期的風險趨避係數

period_daily_returns=daily_returns[:length_of_train*length_of_one_period]
mu_period=period_daily_returns.mean().values*length_of_sub_period
std_period=period_daily_returns.std().values*np.sqrt(length_of_sub_period)
cov_period=period_daily_returns.cov().values*length_of_sub_period
fund_index=select_proper_funds(saa_target[0])

# 篩選掉比重過小的基金並重新建構基金投組
while True:
    Fund_mu,Fund_cov,Fund_cor=funds_returns_cor_cov(fund_index,mu_period,cov_period)
    Fund_std=Fund_cov.diagonal()**0.5
    alpha=alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target[0])
    fund_target_ratio=target_ratio_of_funds_portfolio(alpha,Fund_mu,Fund_cov)
    fund_target_ratio=check_the_real_target(fund_target_ratio,Fund_mu,Fund_cov,alpha)
    if len(np.where(fund_target_ratio<0.001)[0])>0:
        fund_index=np.delete(fund_index,np.where(fund_target_ratio<0.001)[0])
    else:
        break

period_alpha.append(alpha)    
index0=np.copy(fund_index)
#print("距離目標比率最大距離 : %f"%(max(Fund_std)*3))
Fund_target_ratio=np.zeros(unbundle_data.shape[0])
for i in range(len(index0)):
    Fund_target_ratio[index0[i]]=fund_target_ratio[i]
period_fund_target_ratio=np.copy(Fund_target_ratio)

target_expand=[]
for i in range(len(index0)):
    target_expand.append([])
    
target_expand=expand(target_expand,index0,[],[],Fund_mu,Fund_std,fund_target_ratio,np.zeros(len(index0)))
states=states_construction(target_expand)
number_of_states.append(len(states))
print("number of states : %d"%len(states))
final_target_ratio_index=check_the_utility(states,Fund_mu,Fund_cov,alpha)
transition_matrix=transition(states,Fund_mu,Fund_std,Fund_cor)

tStart = time.time()
policy=dynamic_programming(states,transition_matrix,Fund_mu,Fund_cov,final_target_ratio_index,alpha)
tEnd = time.time()
running_time.append(tEnd-tStart)
print("DP花了這麼多時間 : %f\n"%(tEnd-tStart))

weight=np.copy(fund_target_ratio)
period_weight[-1].append(weight.tolist())

for first_3_periods in range(9):

    period_weight.append([])
    after_one_period_data=fund_daily_data[:,index0][first_3_periods*length_of_sub_period:(1+first_3_periods)*length_of_sub_period]
    after_one_period_return=after_one_period_data[-1]/after_one_period_data[0]
    portfolio_growth=np.dot(weight,after_one_period_return)
    weight=weight*after_one_period_return/portfolio_growth
    period_weight[-1].append(weight.tolist())
    
    weight_index=check_weight(weight,states)
    trading_cost,tracking_error,weight=policy_check(weight,weight_index,policy,states,Fund_mu,Fund_cov,final_target_ratio_index)

    trading_cost_list.append(trading_cost)
    tracking_error_list.append(tracking_error)
    period_weight[-1].append(weight.tolist())

# 儲存每一期基金目標比率(這裡每3期改變一次)
for first_3_periods in range(2):
    period_alpha.append(alpha)
    period_fund_target_ratio=np.vstack((period_fund_target_ratio,Fund_target_ratio))
    

for period in range(1,int(len(saa_target)/3)):
    
    print("The %d th time"%period)     
    period_daily_returns=daily_returns[3*period*length_of_one_period:(3*period+length_of_train)*length_of_one_period]
    mu_period=period_daily_returns.mean().values*length_of_sub_period
    std_period=period_daily_returns.std().values*np.sqrt(length_of_sub_period)
    cov_period=period_daily_returns.cov().values*length_of_sub_period
    fund_index=select_proper_funds(saa_target[period*3])
    
    while True:
        Fund_mu,Fund_cov,Fund_cor=funds_returns_cor_cov(fund_index,mu_period,cov_period)
        alpha=alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target[period*3])
        fund_target_ratio=target_ratio_of_funds_portfolio(alpha,Fund_mu,Fund_cov)
        if len(np.where(fund_target_ratio<0.001)[0])>0:
            fund_index=np.delete(fund_index,np.where(fund_target_ratio<0.001)[0])
        else:
            break
    
    index1=np.copy(fund_index)
    union=np.sort(list(set(index0)|set(index1))).tolist() # 上一期和這一期的基金聯集
    Fund_union_mu,Fund_union_cov,Fund_union_cor=funds_returns_cor_cov(union,mu_period,cov_period)
    Fund_union_std=Fund_union_cov.diagonal()**0.5
    alpha=alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target[period*3])
    fund_target_ratio=target_ratio_of_funds_portfolio(alpha,Fund_union_mu,Fund_union_cov)
    print("距離目標比率最大距離 : %f"%(max(Fund_union_std)*3))
    weight,to_be_bought_index,to_be_sold_index=weight_expand(union,index0,index1,weight)
    
    target_expand=[]
    for i in range(len(union)):
        target_expand.append([])
    
    # 建構所有基金可能的權重
    target_expand=expand(target_expand,union,to_be_bought_index,to_be_sold_index,Fund_union_mu,Fund_union_std,fund_target_ratio,weight)
    states=states_construction(target_expand) # 建構所有狀態
    print("union : ")
    print(union)
    print("number of states : %d"%len(states))
    number_of_states.append(len(states))
    
    final_target_ratio_index=check_the_utility(states,Fund_union_mu,Fund_union_cov,alpha)
    Fund_target_ratio=np.zeros(unbundle_data.shape[0])
    for i in range(len(union)):
        Fund_target_ratio[union[i]]=states[final_target_ratio_index][i]
         
    print("target ratio :")
    print(states[final_target_ratio_index])        
    
    transition_matrix=transition(states,Fund_union_mu,Fund_union_std,Fund_union_cor)
    
    tStart = time.time()
    policy=dynamic_programming(states,transition_matrix,Fund_union_mu,Fund_union_cov,final_target_ratio_index,alpha)
    tEnd = time.time()
    running_time.append(tEnd-tStart)
    print("DP 花了那麼多時間 : %f"%(tEnd-tStart))    
    
    # 提取 9 個 sub-period 以後的基金每日價格
    after_3_peroid_data=fund_daily_data[:,union][length_of_one_period*period*3:length_of_one_period*(period+1)*3]
    weight=np.array(weight)
    
    for first_3_periods in range(9):
       
        period_weight.append([])
        print("The origional weight :")
        print(weight.tolist())
        period_weight[-1].append(weight.tolist())
        weight_index=check_weight(weight,states)  # 判斷真實權重離哪個狀態最近
        trading_cost,tracking_error,weight=policy_check(weight,weight_index,policy,states,Fund_union_mu,Fund_union_cov,final_target_ratio_index)
        trading_cost_list.append(trading_cost)
        tracking_error_list.append(tracking_error)
        period_weight[-1].append(weight.tolist())
        print("After rebalancing, the weight becomes :")
        print(weight)
        print("\n")
        print("trading cost : %f"%trading_cost)
        print("tracking error : %f"%tracking_error)
        after_one_period_data=after_3_peroid_data[first_3_periods*length_of_sub_period:(1+first_3_periods)*length_of_sub_period]
        after_one_period_return=after_one_period_data[-1]/after_one_period_data[0]
    
        portfolio_total_growth=sum(weight*after_one_period_return)
        weight=np.array((weight*after_one_period_return/portfolio_total_growth)) # 計算調整過後，再經過一期而改變的權重
        
    for i in range(3):
        period_fund_target_ratio=np.vstack((period_fund_target_ratio,Fund_target_ratio))
        period_alpha.append(alpha)    
    weight_zero=np.where(abs(weight-0)<0.01)[0] # 刪除賣光的基金的權重
    weight=np.delete(weight,weight_zero)
    index0=np.delete(union,weight_zero)
    print("\n")

    
#%%

print("Total tracking error is : %f"%(sum(tracking_error_list)))
print("Total trading cost is : %f"%sum(trading_cost_list))
print("Total cost is : %f"%(sum(tracking_error_list)+sum(trading_cost_list)))       
    
#%%

with open('period_alpha.txt', 'w') as outfile:
   json.dump(period_alpha, outfile)

with open('period_fund_target_ratio.txt', 'w') as outfile:
   json.dump(period_fund_target_ratio.tolist(), outfile)   
    
with open('DP trading cost.txt', 'w') as outfile:
    json.dump(trading_cost_list, outfile)
    
with open('DP tracking error.txt', 'w') as outfile:
    json.dump(tracking_error_list, outfile)    
    
with open('period weight.txt', 'w') as outfile:
    json.dump(period_weight, outfile)  

with open('running time (change every 3 periods).txt', 'w') as outfile:
    json.dump(running_time, outfile) 

with open('number of states(change every 3 periods).txt', 'w') as outfile:
    json.dump(number_of_states, outfile) 

#%%

# with open('DP trading cost.txt') as json_file:
#     trading_cost_list=json.load(json_file)
    
# with open('DP tracking error.txt') as json_file:
#     tracking_error_list=json.load(json_file)   
    
# with open('period weight.txt') as json_file:
#     period_weight=json.load(json_file)

# with open('period_fund_target_ratio.txt') as json_file:
#     period_fund_target_ratio=json.load(json_file)

#%%
# period_fund_target_ratio=np.array(period_fund_target_ratio)
# pp=[]
# for i in range(len(period_fund_target_ratio)):
#     pp.append([])
#     pp[-1].append(np.where(period_fund_target_ratio[i]>0)[0])
#     pp[-1].append(period_fund_target_ratio[i][np.where(period_fund_target_ratio[i]>0)[0]])
#     print(np.where(period_fund_target_ratio[i]>0)[0])
#     print(period_fund_target_ratio[i][np.where(period_fund_target_ratio[i]>0)[0]])
#     print("\n")