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

os.chdir(r"C:\Users\Yoga\Desktop\Auto Rebalance 程式\fund rebalance")

unbundle_data=pd.read_excel(r"C:\Users\Yoga\Desktop\Auto Rebalance 程式\fund rebalance\better unbundle (partial processed).xlsx")
saa_raw_data=pd.read_excel(r"C:\Users\Yoga\Desktop\Auto Rebalance 程式\fund rebalance\1995_2019 daily data.xlsx").iloc[611:]
fund_daily_data=pd.read_excel(r"C:\Users\Yoga\Desktop\Auto Rebalance 程式\fund rebalance\partial daily price of funds.xlsx").drop(["ISIN 代碼"],axis=1).values
#從2007/1/1 開始 #SAA目標比例設定估計期間從2002/3/4開始
with open('period target ratio (change every period).txt') as json_file:
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

num_saa=11
N=price_data.shape[1]

#%%

Fund=np.array(unbundle_data.iloc[:,1:]).T #T是轉置
N=Fund.shape[1] #shape[1]代表第二個維度大小，[0]是第一個維度

def select_proper_funds(saa_target): #一開始用哪幾項基金去逼近SAA
    
    def error(weight): #要最小化的error；調整基金的權重去最小化error
        approximate_saa_weight=np.dot(Fund,weight)
        e=(approximate_saa_weight-saa_target)**2 #e就是歐式距離
        return sum(e)

    def constraint1(weight):
        return float(1)-sum(weight) #限制式:權重和=1

    weight0=np.full(N,1/N) #full是全部都一樣的意思；起始用equal weight
    b=(0,1) #權重範圍是0-1
    bnds=(b,)*N #全部基金的權重
    con1={"type":"eq","fun":constraint1} #"type":"eq"代表限制式是等式
    cons=[con1]

    sol=minimize(error,weight0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18) #目標式,初始值,SLSQP是最小化的方法,邊界,限制式,可以容忍的最小誤差
    fund_index=np.where(sol.x>0.00001)[0] #用where篩選過低的權重

    return fund_index

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

def alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target):
    
    fund_portfolio=Fund[:,fund_index]
    num_fund=len(fund_index)
    alpha_list=np.linspace(1.5,5,351) #窮舉所有alpha(效用函數中的風險趨避程度)
    error_test=[]
    
    def Utility(weight):
        return -np.dot(weight,Fund_mu)+0.5*alpha_list[i]*np.dot(weight.T,np.dot(Fund_cov,weight)) #因為套件本身是極小化，所以加-號

    def constraint1(weight):
        return 1-sum(weight)
    
    weight0=np.full(num_fund,1/num_fund)
    b=(0,1)
    bnds=(b,)*num_fund
    con1={"type":"eq","fun":constraint1}
    cons=[con1]
    
    for i in range(len(alpha_list)):
        sol=minimize(Utility,weight0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18) #每個alpha都算一組解
        error_test.append(sum((np.dot(fund_portfolio,sol.x)-saa_target)**2))
    
    return np.round(alpha_list[np.argmin(error_test)],2) #argmin回傳最小值的位置；2是到小數第二位

def target_ratio_of_funds_portfolio(alpha,Fund_mu,Fund_cov): #和select_proper_funds幾乎相同
    
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

def check_the_real_target(fund_target_ratio,Fund_union_mu,Fund_union_cov,alpha):
    
    def Utility(weight):
        return np.dot(weight,Fund_union_mu)-0.5*alpha*np.dot(weight.T,np.dot(Fund_union_cov,weight))
    
    precision=10**-8
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
    

def weight_expand(union,index0,index1,weight): 
    
    weight=weight.tolist()
    to_be_bought_index=np.sort(list(set(union)-set(index0))) #聯集-上一期=這一期新買的
    to_be_sold_index=np.sort(list(set(union)-set(index1))) #聯集-這一期=這一期賣掉的
    for i in to_be_bought_index:
        weight.insert(union.index(i),0)  #如果有新買的話要擴充上一期的陣列，insert是直接在index的位置插入0；賣掉的不用改
    
    return weight,to_be_bought_index,to_be_sold_index
    
#expand﹔展開所有可能的權重，包含上下界
def expand(target_expand,union,to_be_bought_index,to_be_sold_index,Fund_union_mu,Fund_union_std,fund_target_ratio,weight):
    
    k=3 #最大範圍在k個標準差
    inter=min(Fund_union_std)*2/3 #inter是區間往外/內長的時候一次要漲多少
    be_bought,be_sold,all_up,all_down,up,down,interval=[],[],[],[],[],[],[]
    length_index=len(union) #總共選到的基金數量
    
    for i in to_be_sold_index:
        be_sold.append(union.index(i)) #存被賣掉的基金index
    for i in to_be_bought_index:
        be_bought.append(union.index(i))
    
    for i in range(length_index):
        all_up.append(1+Fund_union_mu[i]+k*Fund_union_std[i]) #all_up代表地i檔基金上漲最多的幅度
        all_down.append(1+Fund_union_mu[i]-k*Fund_union_std[i])
    for i in range(length_index):
        up.append(fund_target_ratio[i]*all_up[i]/(np.dot(fund_target_ratio,all_down)+2*k*Fund_union_std[i])) #up是基金的上界=權重最大的狀況(只有自己漲，其他人都跌，這樣權重和才會是1)
        down.append(fund_target_ratio[i]*all_down[i]/(np.dot(fund_target_ratio,all_up)-2*k*Fund_union_std[i]))
    for i in range(length_index):
        interval.append([])
    for i in range(length_index):
        interval[i].append(down[i])
        interval[i].append(up[i])   
    
    for i in range(length_index): #因為我們的上下界之間一定會包含目標比率，所以先把目標比率先放進去
        target_expand[i].append(fund_target_ratio[i])  #target_expand是基金可能跑到的權重
    
    for i in range(length_index): #分三種case:這期要多買的、這期要多賣的、這期調整原本有的
        if i in be_bought:
            while target_expand[i][-1]<interval[i][1]: #target_expand[i][-1]是上界;因為interval只存2個值，所以[1]是上界，[0]是下界
                target_expand[i].append(target_expand[i][-1]+inter)
            while target_expand[i][0]-inter>0: #買的下界是0，所以可以再往下長
                target_expand[i].insert(0,target_expand[i][0]-inter)
            target_expand[i].insert(0,0) #因為可能不買，所以插入0
        elif i in be_sold:  #因為是賣的case，所以不會有往上長的情況，只考慮往下長
            while target_expand[i][-1]+inter<interval[i][1] or target_expand[i][-1]+inter<weight[i]:
                target_expand[i].append(target_expand[i][-1]+inter)
        else:
            if fund_target_ratio[i]>weight[i]: #target_ratio>當前的比例
                while target_expand[i][-1]+inter<interval[i][1]:
                    target_expand[i].append(target_expand[i][-1]+inter)
                while target_expand[i][0]-inter>interval[i][0] or target_expand[i][0]>weight[i]: #interval[i][0]是區間下界；因為區間一定要包含到我目前的比例，所以沒碰到之前要往下長
                    target_expand[i].insert(0,target_expand[i][0]-inter)
            else:
                if fund_target_ratio[i]<weight[i]:
                    while target_expand[i][-1]+inter<interval[i][1] or target_expand[i][-1]<weight[i]:
                        target_expand[i].append(target_expand[i][-1]+inter)
                    while target_expand[i][0]-inter>interval[i][0]:
                        target_expand[i].insert(0,target_expand[i][0]-inter)
    
    return target_expand
    
def states_construction(target_expand):  #expend做完後再用這個找出權重和為1的組合
    states=np.array(list(product(*target_expand))) #product排列組合所有可能
    states=states[abs(np.sum(states,1)-1)<0.0001] #只取出權重和=1的state；abs(np.sum(states,1)-1)代表以row加總，也就是權重和
    return states

def check_the_utility(states,union_return,union_cov,alpha):
    
    def Rce(j):
        return np.dot(union_return,states[j])-0.5*alpha*np.dot(states[j].T,np.dot(union_cov,states[j]))
    
    test=[]
    for i in range(len(states)): #states就是權重
        test.append(Rce(i))
    
    return np.argmax(test) #argmax回傳效用最大的states

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
        for j in range(N):
            growth[i][j]=np.exp((union_return[j]-0.5*union_std[j]**2)+union_std[j]*A[i][j])
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
        
def dynamic_programming(states,transition_matrix,union_return,union_cov,final_target_ratio_index,alpha):
    
    beta=np.exp(-0.02/12) #bellman equation的折現值
    #beta=0.95
    ite=15  #policy_evaluation的迭代次數
    number_of_states=len(states)
    policy_PI=np.zeros(number_of_states)  #在該位置(當下權重)要用的策略
    J=np.zeros(number_of_states)  #存放每個策略的成本
    
    def Rce(j):
        return np.dot(union_return,states[j])-0.5*alpha*np.dot(states[j].T,np.dot(union_cov,states[j]))
    
    def trading_cost(i,j): #從i到j的交易成本
        dim=len(union_return)
        C=np.zeros(dim)
        for k in range(dim):
            if states[j][k]-states[i][k]>0:
                C[k]=0.015 # buy的手續費是1.5%
            else:
                C[k]=0 #sell不用手續費
        return np.dot(C,abs(states[i]-states[j])) #C存的是在那個位置是買還是賣；內積後就是成本
    
    Rce_target=Rce(final_target_ratio_index) #target_ratio的效用
    def tracking_error(j):
        return Rce_target-Rce(j) #把target_ratio的效用減掉調整到j的效用
    
    first_policy=np.zeros(number_of_states) #迭代的初始值
    for i in range(number_of_states):
        first_policy[i]=final_target_ratio_index #一開始我的目標就是先調整到target_ration
    
    policy_PI=np.vstack((policy_PI,first_policy))
    
    def policy_evaluation(k): #某個策略的期望總成本，k是index
        J=np.zeros((ite+1,number_of_states)) #因為第一行的迭代全部都是0，所以迭代次數要+1
        for i in range(1,ite):
            for j in range(number_of_states):
                J[i][j]=trading_cost(j,k[j])+tracking_error(k[j])+beta*np.dot(transition_matrix[k[j]],J[i-1]) #第i次迭代的第j個狀態；k[j]:在j位置所使用的策略
        return J[-1]

    def policy_improvement(i,J):  #找出期望總成本最小的策略
        compare=np.zeros(number_of_states)
        for k in range(number_of_states):
            compare[k]= trading_cost(i,k)+tracking_error(k)+beta*np.dot(transition_matrix[k],J) #調整到各權重的總成本
        policy_PI[-1][i]=np.argmin(compare) #policy_PI[-1]:最新的策略
        return min(compare)
    
    tStart=time.time()
    while ~(policy_PI[-1]==policy_PI[-2]).all() : #當策略收斂就停止
        policy_PI=np.vstack((policy_PI,np.zeros(number_of_states)))
        J=policy_evaluation(policy_PI[-2].astype(int)) #astype(int)轉成整數型態
        for i in range(number_of_states):
            policy_improvement(i,J)
    tEnd=time.time()
    running=tEnd-tStart
    
    return policy_PI[-1],running

def check_weight(weight,states):
    dis=[]
    number_of_assets=len(states)
    for i in range(number_of_assets):
        dis.append(sum((weight-states[i])**2))
    weight_index=np.argmin(dis)
    return weight_index

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
period_weight=[[]] #二維陣列
Fund=np.array(unbundle_data.iloc[:,1:]).T
length_of_one_period=21
length_of_train=120
trading_cost_list,tracking_error_list=[],[]
period_alpha=[]

period_daily_returns=daily_returns[:length_of_train*length_of_one_period] #SAA的日報酬；120期*21天
mu_period=period_daily_returns.mean().values*length_of_one_period #算一期的
std_period=period_daily_returns.std().values*np.sqrt(length_of_one_period)
cov_period=period_daily_returns.cov().values*length_of_one_period
fund_index=select_proper_funds(saa_target[0]) #0就是第一期
# 以下開始alpha calibration
while True:
    Fund_mu,Fund_cov,Fund_cor=funds_returns_cor_cov(fund_index,mu_period,cov_period)
    alpha=alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target[0]) #在效用最大化的狀況下，找出能使基金目標比例最接近SAA-target的alpha
    fund_target_ratio=target_ratio_of_funds_portfolio(alpha,Fund_mu,Fund_cov)
    if len(np.where(fund_target_ratio<0.001)[0])>0: #>0代表有值所以刪除，刪除過小的index
        fund_index=np.delete(fund_index,np.where(fund_target_ratio<0.001)[0])
    else:
        break #跑到不會有過小值為止
    
period_alpha.append(alpha)    
index0=np.copy(fund_index)

Fund_target_ratio=np.zeros(unbundle_data.shape[0]) #第一個維度是基金的數量
for i in range(len(index0)): 
    Fund_target_ratio[index0[i]]=fund_target_ratio[i]
period_fund_target_ratio=np.copy(Fund_target_ratio)

#print(np.dot(Fund[:,fund_index],fund_target_ratio))

after_one_period_data=fund_daily_data[:,index0][:length_of_one_period]
after_one_period_return=after_one_period_data[length_of_one_period-1]/after_one_period_data[0] #只算一期

portfolio_growth=np.dot(fund_target_ratio,after_one_period_return) #原本的權重*報酬率
weight=fund_target_ratio*after_one_period_return/portfolio_growth #*是對應位置相乘；矩陣乘法要用dot
period_weight[-1].append(weight.tolist()) #因為是二維陣列，所以-1代表最後一個row；tolist轉成list
#len(saa_target)

for period in range(1,len(saa_target)): #算出每一期基金的target_ratio，因為第一期上面算完了，所以從1開始
    print("The %d th time"%period)
    period_weight.append([])     
    period_daily_returns=daily_returns[period*length_of_one_period:(period+length_of_train)*length_of_one_period]
    mu_period=period_daily_returns.mean().values*length_of_one_period
    std_period=period_daily_returns.std().values*np.sqrt(length_of_one_period)
    cov_period=period_daily_returns.cov().values*length_of_one_period
    fund_index=select_proper_funds(saa_target[period])
    
    while True:
        Fund_mu,Fund_cov,Fund_cor=funds_returns_cor_cov(fund_index,mu_period,cov_period)
        alpha=alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target[period])
        fund_target_ratio=target_ratio_of_funds_portfolio(alpha,Fund_mu,Fund_cov)
        if len(np.where(fund_target_ratio<0.001)[0])>0:
            fund_index=np.delete(fund_index,np.where(fund_target_ratio<0.001)[0])
        else:
            break
    
    index1=np.copy(fund_index)
    union=np.sort(list(set(index0)|set(index1))).tolist() #只要是兩期有出現過的就留下來；set變成集合；|是聯集:轉成list後sort排序
    Fund_union_mu,Fund_union_cov,Fund_union_cor=funds_returns_cor_cov(union,mu_period,cov_period)
    Fund_union_std=Fund_union_cov.diagonal()**0.5 #標準差是對角線開根號
    alpha=alpha_calibration(fund_index,Fund_mu,Fund_cov,saa_target[period])
    period_alpha.append(alpha)
    fund_target_ratio=target_ratio_of_funds_portfolio(alpha,Fund_union_mu,Fund_union_cov)
    
    weight,to_be_bought_index,to_be_sold_index=weight_expand(union,index0,index1,weight) #to_be_bought_index上一期沒有這期有；to_be_sold_index上祇有這期沒有
    
    target_expand=[]
    for i in range(len(union)):
        target_expand.append([]) #存區間的上下界
 
    target_expand=expand(target_expand,union,to_be_bought_index,to_be_sold_index,Fund_union_mu,Fund_union_std,fund_target_ratio,weight) #weight是調整前的權重
    states=states_construction(target_expand)
    
    print("union : ")
    print(union)
    print("number of states : %d"%len(states))
    print("The origional weight :")
    print(weight)   
    period_weight[-1].append(weight)
    number_of_states.append(len(states))
    
    final_target_ratio_index=check_the_utility(states,Fund_union_mu,Fund_union_cov,alpha)
    Fund_target_ratio=np.zeros(unbundle_data.shape[0])
    for i in range(len(union)):
        Fund_target_ratio[union[i]]=states[final_target_ratio_index][i]
        
    period_fund_target_ratio=np.vstack((period_fund_target_ratio,Fund_target_ratio)) #vstack是np.array的增加方法
    print("target ratio :")
    print(states[final_target_ratio_index])        
    
    transition_matrix=transition(states,Fund_union_mu,Fund_union_std,Fund_union_cor)
    

    policy,running=dynamic_programming(states,transition_matrix,Fund_union_mu,Fund_union_cov,final_target_ratio_index,alpha)
    running_time.append(running)
    print("DP 花了那麼多時間 : %f"%running)
    
    weight_index=check_weight(weight,states)
    trading_cost,tracking_error,weight=policy_check(weight,weight_index,policy,states,Fund_union_mu,Fund_union_cov,final_target_ratio_index)
    trading_cost_list.append(trading_cost)
    tracking_error_list.append(tracking_error)
    period_weight[-1].append(weight.tolist())
    print("After rebalancing, the weight becomes :")
    print(weight)
    print("trading cost : %f"%trading_cost)
    print("tracking error : %f"%tracking_error)
    
    after_one_period_data=fund_daily_data[:,union][length_of_one_period*period:length_of_one_period*(period+1)]
    after_one_period_return=after_one_period_data[-1]/after_one_period_data[0]
    
    portfolio_total_growth=sum(weight*after_one_period_return)
    weight=np.array((weight*after_one_period_return/portfolio_total_growth))
    
    weight_zero=np.where(abs(weight-0)<0.01)[0]
    weight=np.delete(weight,weight_zero)
    index0=np.delete(union,weight_zero)
    print("\n")

#%%
print("Total tracking error is : %f"%(sum(tracking_error_list)))
print("Total trading cost is : %f"%sum(trading_cost_list))
print("Total cost is : %f"%(sum(tracking_error_list)+sum(trading_cost_list)))       
    
#%%

with open('period_alpha (change every period).txt', 'w') as outfile:
   json.dump(period_alpha, outfile)

with open('period_fund_target_ratio (change every period).txt', 'w') as outfile:
   json.dump(period_fund_target_ratio.tolist(), outfile)   
    
with open('DP trading cost (change every period).txt', 'w') as outfile:
    json.dump(trading_cost_list, outfile)
    
with open('DP tracking error (change every period).txt', 'w') as outfile:
    json.dump(tracking_error_list, outfile)    
    
with open('period weight (change every period).txt', 'w') as outfile:
    json.dump(period_weight, outfile)  
    
with open('running time.txt', 'w') as outfile:
    json.dump(running_time, outfile) 

with open('number of states.txt', 'w') as outfile:
    json.dump(number_of_states, outfile) 
    
#%%

# with open('DP trading cost (change every period).txt') as json_file:
#     trading_cost_list=json.load(json_file)
    
# with open('DP tracking error (change every period).txt') as json_file:
#     tracking_error_list=json.load(json_file)   
    
# with open('period weight (change every period).txt') as json_file:
#     period_weight=json.load(json_file)

# with open('period_fund_target_ratio (change every period).txt') as json_file:
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
    
#%%
# for i in range(1,len(period_weight)):
#     if period_weight[i][0]!=period_weight[i][1]:
#         print(i)
#         print(np.nonzero(period_fund_target_ratio[i])[0])
#         print(period_weight[i][0])
#         print(period_weight[i][1])
#         print("\n")