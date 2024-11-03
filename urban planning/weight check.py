import numpy as np
import pandas as pd
import json
import os


os.chdir(r"C:\Users\bb19x\OneDrive\桌面\何もない\Bellman Euqtion new SAA\diefferent state selection\urban planning 4 saa")

with open('final weight states.txt') as json_file:
    com = json.load(json_file)

with open('approx policy (PI).txt') as json_file:
    policy = json.load(json_file)
policy=np.array(policy)

with open('daily returns.txt') as json_file:
    returns = json.load(json_file)

with open('expected annual return.txt') as json_file:
    mu = json.load(json_file)

with open('target.txt') as json_file:
    target = json.load(json_file)

newSAA=pd.read_excel("C:/Users/bb19x/OneDrive/桌面/何もない/Bellman Euqtion new SAA/csv_data/2000_2019 daily data.xlsx")
newSAA=newSAA.drop(["Dates"],axis=1)

tickers=[]

for i in newSAA:
    if i=="Dates":
        pass
    else:
        tickers.append(i)

asset_period_index=pd.DataFrame(index=np.arange(0,int(newSAA.shape[0]/21)),columns=tickers)
for i in range(0,int(newSAA.shape[0]/21)):
    asset_period_index.iloc[i]=newSAA.iloc[21*i]
    
#%%

test_period=pd.DataFrame()
test_period["SHCOMP Index"]=asset_period_index["SHCOMP Index"]
test_period["SENSEX Index"]=asset_period_index["SENSEX Index"]
test_period["MXLA Index"]=asset_period_index["MXLA Index"]
test_period["SPX Index"]=asset_period_index["SPX Index"]

tickers = []
for i in test_period:
    tickers.append(i)

asset_period_return=pd.DataFrame()

for t in tickers:
    if t=="Dates": continue
    asset_period_return[t]=asset_period_index[t]/asset_period_index[t].shift(1)-1

N=asset_period_return.shape[1]
dt=1/12
alpha=3
cov=pd.DataFrame(returns).cov()*252
C_buy=0.001425
C_sell=0.004425

target_ratio=np.array(target)
period_return=np.array(asset_period_return)

def utility(w):
    return np.dot(mu,w)-0.5*alpha*np.dot(w.T,np.dot(cov,w))

def TE(w):
    return (utility(np.array(target_ratio))-utility(w))*dt

def TC(w1,w2):
    C=[]
    for k in range(N):
        if w2[k]-w1[k]>0:
            C.append(C_buy)
        else:
            C.append(C_sell)
    return np.dot(np.array(C),abs(w2-w1))
	#%%
#Dynamic Programming
weights=np.zeros((asset_period_return.shape[0],N))
tracking_error=[]
trading_cost=[]
weights[0]=target_ratio

for i in range(1,weights.shape[0]):
    growth=np.dot(weights[i-1],(1+period_return[i]))
    weights[i]=weights[i-1]*(1+period_return[i])/growth
    tempt=np.copy(weights[i])
    test=[]
    for k in range(len(com)): #尋找與新的權重最接近的格子點
        test.append(sum((weights[i][l]-com[k][l])**2 for l in range(N)))
    if np.argmin(test)!=int(policy[-1][np.argmin(test)]): #調整OR不調整
        weights[i]=com[int(policy[-1][np.argmin(test)])]
        tracking_error.append(TE(weights[i]))
        trading_cost.append(TC(np.array(tempt),weights[i]))
    else:
        tracking_error.append(TE(weights[i]))
        trading_cost.append(0)
         
  
print("20年間追蹤誤差成本 : %f"%sum(tracking_error))
print("20年間交易成本 : %f"%sum(trading_cost))
print("20年間總成本 : %f"%(sum(tracking_error)+sum(trading_cost)))

print("次貸危機期間追蹤誤差成本 : %f"%sum(tracking_error[92:111]))
print("次貸危機期間交易成本 : %f"%sum(trading_cost[92:111]))
print("次貸危機期間總成本 : %f"%(sum(tracking_error[92:111])+sum(trading_cost[92:111])))

print("共調整 %d 次"%sum(np.array(trading_cost)!=0))
print("次貸危機期間共調整 %d 次"%sum(np.array(trading_cost[92:111])!=0))

