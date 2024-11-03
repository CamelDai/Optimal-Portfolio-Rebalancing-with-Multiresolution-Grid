"""
Created on Wed Oct 30 11:28:39 2019

@author: User
"""
import numpy as np
import pandas as pd
import time
import json
import os
from scipy.optimize import minimize

os.chdir(r"路徑") #更改路徑
#讀取SAA日價格資料
daily_in_sample=pd.read_excel("C:/Users/bb19x/OneDrive/桌面/何もない/Bellman Euqtion new SAA/csv_data/1990_1999 real daily data.xlsx")
#建立新的dataframe
price_data=pd.DataFrame()

#挑以下4檔SAA做實驗
#price_data["G0Q0_Index"]=daily_in_sample["G0Q0 Index"]
#price_data["G0BC_Index"]=daily_in_sample["G0BC Index"]
#price_data["H0A0_Index"]=daily_in_sample["H0A0 Index"]
#price_data["IP00_Index"]=daily_in_sample["IP00 Index"]
#price_data["LDMP_Index"]=daily_in_sample["LDMP Index"]
#price_data["SXXP_Index"]=daily_in_sample["SXXP Index"]
#price_data["NKY_Index"]=daily_in_sample["NKY Index"]
price_data["SHCOMP_Index"]=daily_in_sample["SHCOMP Index"]
#price_data["HSCEI_Index"]=daily_in_sample["HSCEI Index"]
#price_data["TWSE_Index"]=daily_in_sample["TWSE Index"]
#price_data["KOSPI_Index"]=daily_in_sample["KOSPI Index"]
price_data["SENSEX_Index"]=daily_in_sample["SENSEX Index"]
#price_data["MXSO_Index"]=daily_in_sample["MXSO Index"]
#price_data["MXMU_Index"]=daily_in_sample["MXMU Index"]
price_data["MXLA_Index"]=daily_in_sample["MXLA Index"]
price_data["SPX_Index"]=daily_in_sample["SPX Index"]

#計算日報酬
tickers=[]

for i in price_data:
    if i=="Dates":
        pass
    else:
        tickers.append(i)
    
daily_returns = pd.DataFrame()

for t in tickers:
    daily_returns[t] = price_data[t]/price_data[t].shift(1)-1


N=price_data.shape[1] # 投資組合資產個數
alpha=3 # 風險趨避係數
mu_daily=daily_returns.mean().values # 日期望報酬
std_daily=daily_returns.std().values # 日樣本波動度
cov_daily=daily_returns.cov().values # 日共變異數矩陣
cor=daily_returns.corr().values # 相關係數矩陣

mu_annually=daily_returns.mean().values*252 # 年期望報酬
std_annually=daily_returns.std().values*np.sqrt(252) # 年波動度
cov_annually=daily_returns.cov().values*252 # 年共變異數矩陣


#計算SAA目標比例
def Utility(weight):
    return -np.dot(weight,mu_annually)+0.5*alpha*np.dot(weight.T,np.dot(cov_annually,weight))

def constraint1(weight):
    return 1-sum(weight)

weight0=np.full(N,1/N) # 設初始值
b=(0,1) # 投組權重位於0、1之間\
bnds=(b,)*N # N 個 bound
con1={"type":"eq","fun":constraint1} # 限制式1式個等式
cons=[con1]

sol=minimize(Utility,weight0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18)
print(sol.x)
print("Return: %f"%(np.dot(sol.x,mu_annually)))
print("Volatility: %f"%(np.dot(sol.x.T,np.dot(cov_annually,sol.x))))
target=sol.x

#%%
#處理上下界
k=3
dt=1/12 # 一期一個月，所以dt=1/12
Mu=mu_annually*dt # 月期望報酬
std=std_annually*dt**0.5 # 月波動度

up,down=[],[]
all_up,all_down=[],[]

for i in range(N):
    all_up.append(1+Mu[i]+k*std[i])
    all_down.append(1+Mu[i]-k*std[i])
    
for i in range(N):
    up.append(target[i]*(1+Mu[i]+k*std[i])/(np.dot(target,all_down)+2*k*target[i]*std[i]))
    down.append(target[i]*(1+Mu[i]-k*std[i])/(np.dot(target,all_up)-2*k*target[i]*std[i]))

interval=[[down[0],up[0]],[down[1],up[1]],[down[2],up[2]],[down[3],up[3]]]
target_expand=[[target[0]],[target[1]],[target[2]],[target[3]]]

k=0.01 # 以 k 為間距離散狀態空間

for i in range(N):
    while target_expand[i][-1]+k<interval[i][1]:
        target_expand[i].append(target_expand[i][-1]+k)
    while target_expand[i][0]-k>interval[i][0]:
        target_expand[i].insert(0,target_expand[i][0]-k)

#%%

# 建構狀態
com=[]

for a in target_expand[0]:
    for b in target_expand[1]:
        for c in target_expand[2]:
            for d in target_expand[3]:
                if abs(a+b+c+d-1)<0.001:
                    com.append([a,b,c,d])

#com_after矩陣表示第i個狀態經過一期會跑到第j個狀態的次數
com_after=np.zeros((len(com),len(com)))

#%%
#蒙地卡羅 造維度為simulation的向量，每個元素為NX1的normal(0,1)
np.random.seed(123)
simulation=50000
rand=np.random.normal(0,1,(simulation,N)) # rand 為 simulation x N 的矩陣，每個矩陣元素符合 N(0,1) 且 iid
B=np.linalg.cholesky(cor) # Cholesky decomposition

#透過Cholesky Decomposition將取出的隨機變數轉換
A=np.zeros((simulation,N)) # A 用來儲存轉換後的樣本
        
for i in range(simulation):
    A[i]=np.dot(B,rand[i])


#蒙地卡羅
#模擬未來各指數可能的走勢
#com_temp的size為len(com)*simulation
com=np.array(com)
com_temp=np.zeros((simulation*len(com),N))   
growth=np.zeros((simulation,N))

for i in range(simulation):
    growth[i]=np.exp((Mu-0.5*std**2)*dt+np.sqrt(dt)*std*A[i]) # 模擬未來價格漲幅程度
        
tStart = time.time()   
for i in range(len(com)):
    for j in range(simulation):
        com_temp[simulation*i+j]=com[i]*growth[j] # 模擬未來價格路徑
tEnd = time.time()

print("It cost %f sec" % (tEnd - tStart))

#將新的價值轉換為權重
com_temp_weight=np.zeros((len(com_temp),N))

#com_temp_weight的size為len(com)*simulation
for i in range(0,len(com_temp)):
    com_temp_weight[i]=com_temp[i]/sum(com_temp[i]) # 將新模擬的投組價值權重化
        
#將新權重的list編寫為list of list
#將不同權重的模擬切割開來
#weight_after的size為 len(com)

weight_after=np.zeros((len(com),simulation,N))

for i in range(len(np.zeros((len(com),simulation,N)))):
    weight_after[i]=com_temp_weight[i*simulation:(1+i)*simulation]

#weight_after儲存的是每一種權重模擬simulation次，看最後會變成什麼樣權重的結果
#這部分其實就是在分類模擬後的結果應該要歸到哪一種權重
#有種投票的概念，每次模擬都有一票的資格

tStart = time.time()#計時開始

for i in range(len(com)):
    for j in range(simulation):
        dis=np.sum((np.tile(weight_after[i][j],(len(com),1))-com)**2,axis=1)
        com_after[i][np.argmin(dis)]+=1

tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))

#計算機率轉換矩陣
Transition=np.zeros((len(com),len(com)))
for i in range(len(com)):
    Transition[i]=com_after[i]/simulation
#%%
#將結果儲存

with open('transition probability.txt', 'w') as outfile:
   json.dump(Transition.tolist(), outfile)  

with open('daily returns.txt', 'w') as outfile:
   json.dump(daily_returns.values.tolist(), outfile)

with open('expected annual return.txt', 'w') as outfile:
   json.dump(mu_annually.tolist(), outfile)

with open('weight states.txt', 'w') as outfile:
   json.dump(com.tolist(), outfile)

with open('target.txt', 'w') as outfile:
   json.dump(target.tolist(), outfile)