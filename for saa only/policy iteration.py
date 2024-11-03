import numpy as np
import pandas as pd
import json
import os
import time

os.chdir(r"路徑")
#%%輸入資料

# 讀取建構好的狀態
with open('weight states.txt') as json_file:
    com = json.load(json_file)
    
# 提取建構好的轉換機率矩陣
with open('transition probability.txt') as json_file:
    Transition = json.load(json_file)
Transition=np.array(Transition)

# 提取儲存的資產每日報酬
with open('daily returns.txt') as json_file:
    returns = json.load(json_file)

# 提取個資產年期望報酬
with open('expected annual return.txt') as json_file:
    mu = json.load(json_file)
mu=np.array(mu)

# 提取 SAA 目標比率
with open('target.txt') as json_file:
    target = json.load(json_file)

#%%
#參數設置 
N=len(mu) # 投組資產數
dt=1/12
mu=mu*dt # 月化年期望報酬
returns=pd.DataFrame(returns)
cov=(returns.cov()*252*dt).values # 月化的共變異數矩陣
alpha=3 # 風險趨避係數
C_buy=0.001425 # 買進成本
C_sell=0.004425 # 賣出成本
length_com=len(com) # 狀態個數

target=com.index(target) # 看SAA目標比率是在所有狀態中第幾號狀態
beta=np.exp(-0.02/12) # 折現因子
#beta=0.9

#policy_PI為用來儲存每一次迭代的最佳策略
policy_PI=np.zeros(length_com)

com=np.array(com)

#%%
#Certainty Equivalence Return
def Rce(j):
    return np.dot(mu,com[j])-0.5*alpha*np.dot(com[j].T,
                  np.dot(cov,com[j]))

#Transaction Cost
def trading_cost(i,j):
    C=np.zeros(N)
    for k in range(N):
        if com[j][k]-com[i][k]>0:
            C[k]=C_buy
        else:
            C[k]=C_sell
    return np.dot(C,abs(com[i]-com[j]))

#Tracking Error
Rce_target=Rce(target)
def tracking_error(j):
    return Rce_target-Rce(j)

#%%
#將每個狀態的初始決策設為調整到目標比率
first_policy=np.zeros(length_com)
for i in range(length_com):
    first_policy[i]=target
    
policy_PI=np.vstack((policy_PI,first_policy))        
    
#%%
ite=15

def policy_evaluation(k):
    J=np.zeros((ite+1,length_com))
    for i in range(1,ite+1):
        for j in range(length_com):
            J[i][j]=trading_cost(j,k[j])+tracking_error(k[j])+beta*np.dot(Transition[k[j]],J[i-1])
    return J[-1]


def policy_improvement(i,J):
    compare=np.zeros(length_com)
    for k in range(length_com):
        compare[k]= trading_cost(i,k)+tracking_error(k)+beta*np.dot(Transition[k],J)
    policy_PI[-1][i]=np.argmin(compare)


#%%
#迭代
tStart = time.time() 

while ~(policy_PI[-1]==policy_PI[-2]).all() :
    policy_PI=np.vstack((policy_PI,np.zeros(length_com)))
    J=policy_evaluation(policy_PI[-2].astype(int))
    for i in range(length_com):
        policy_improvement(i,J)
    
tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))

#%%
#查看無交易區間
        
no_trade_region=[]

for i in range(len(com)):
    if policy_PI[-1][i]==i:
        no_trade_region.append([com[i].tolist()])        


#%%

with open('policy (PI).txt', 'w') as outfile:
   json.dump(policy_PI.tolist(), outfile)

with open('No Trade Region.txt', 'w') as outfile:
   json.dump(no_trade_region, outfile)

with open('J table.txt', 'w') as outfile:
   json.dump(J.tolist(), outfile)



