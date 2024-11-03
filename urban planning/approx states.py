import numpy as np
import os
import json
#from pulp import *
from scipy.optimize import minimize
import pandas as pd

os.chdir(r"C:\Users\bb19x\OneDrive\桌面\何もない\Bellman Euqtion new SAA\diefferent state selection\urban planning 4 saa")

    
data_raw=pd.read_excel("C:/Users/bb19x/OneDrive/桌面/何もない/Bellman Euqtion new SAA/csv_data/1990_1999 real daily data.xlsx")    
daily=data_raw[:-1]
daily_in_sample=pd.DataFrame(daily)   
price_data=pd.DataFrame()

price_data["SHCOMP_Index"]=daily_in_sample["SHCOMP Index"]
price_data["SENSEX_Index"]=daily_in_sample["SENSEX Index"]
price_data["MXLA_Index"]=daily_in_sample["MXLA Index"]
price_data["SPX_Index"]=daily_in_sample["SPX Index"]

tickers=[]

for i in price_data:
    if i=="Dates":
        pass
    else:
        tickers.append(i)
        
daily_returns = pd.DataFrame()
for t in tickers:
    daily_returns[t] = price_data[t]/price_data[t].shift(1)-1


N=price_data.shape[1]
alpha=3
mu_daily=daily_returns.mean().values
std_daily=daily_returns.std().values
cov_daily=daily_returns.cov().values
cor=daily_returns.corr().values

mu_annually=daily_returns.mean().values*252
std_annually=daily_returns.std().values*np.sqrt(252)
cov_annually=daily_returns.cov().values*252

with open('growth.txt') as json_file:
    growth = json.load(json_file)
growth=np.array(growth)

with open('target.txt') as json_file:
    target = json.load(json_file)
target=np.array(target)

with open('interval.txt') as json_file:
    interval = json.load(json_file)
#%%

#尋找SAA目標比例
# def Utility(weight):
#     return -np.dot(weight,mu_annually)+0.5*alpha*np.dot(weight.T,np.dot(cov_annually,weight))

# def constraint1(weight):
#     return 1-sum(weight)

# weight0=np.full(N,1/N)
# b=(0,1)
# bnds=(b,)*N
# con1={"type":"eq","fun":constraint1}
# cons=[con1]

# sol=minimize(Utility,weight0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18)
# print(sol.x)
# print("Return: %f"%(np.dot(sol.x,mu_annually)))
# print("Volatility: %f"%(np.dot(sol.x.T,np.dot(cov_annually,sol.x))))

# target=sol.x

#%%
# k=3
# dt=1/12
# Mu=mu_annually*dt
# std=std_annually*dt**0.5

# up,down=[],[]
# all_up,all_down=[],[]

# for i in range(N):
#     all_up.append(1+Mu[i]+k*std[i])
#     all_down.append(1+Mu[i]-k*std[i])
    
# for i in range(N):
#     up.append(target[i]*(1+Mu[i]+k*std[i])/(np.dot(target,all_down)+2*k*target[i]*std[i]))
#     down.append(target[i]*(1+Mu[i]-k*std[i])/(np.dot(target,all_up)-2*k*target[i]*std[i]))

# interval=[[down[0],up[0]],[down[1],up[1]],[down[2],up[2]],[down[3],up[3]]]

# simulation=50000
# rand=np.random.normal(0,1,(simulation,N))
# B=np.linalg.cholesky(cor)

#透過Cholesky Decomposition將取出的隨機變數轉換
# A=np.zeros((simulation,N))
        
# for i in range(simulation):
#     A[i]=np.dot(B,rand[i])

# growth=np.zeros((simulation,N))

#%%
# for i in range(simulation):
#     growth[i]=np.exp((Mu-0.5*std**2)*dt+np.sqrt(dt)*std*A[i])

# with open('growth.txt', 'w') as outfile:
#     json.dump(growth.tolist(), outfile)

# with open('daily returns.txt', 'w') as outfile:
#     json.dump(daily_returns.values.tolist(), outfile)

# with open('target.txt', 'w') as outfile:
#     json.dump(target.tolist(), outfile)

# with open('expected annual return.txt', 'w') as outfile:
#     json.dump(mu_annually.tolist(), outfile)

# with open('interval.txt', 'w') as outfile:
#     json.dump(interval, outfile)
    
#%%
    
#以k%來切割區塊

approx_target_expand=[[target[0]],[target[1]],[target[2]],[target[3]]]
approx_com=[]
simulation=len(growth)
delta=0.02

for i in range(N):
    while approx_target_expand[i][-1]+delta<interval[i][1]:
        approx_target_expand[i].append(approx_target_expand[i][-1]+delta)
    while approx_target_expand[i][0]-delta>interval[i][0]:
        approx_target_expand[i].insert(0,approx_target_expand[i][0]-delta)

for a in approx_target_expand[0]:
    for b in approx_target_expand[1]:
        for c in approx_target_expand[2]:
            for d in approx_target_expand[3]:
                if abs(a+b+c+d-1)<0.0001:
                    approx_com.append([a,b,c,d])
#%%
#計算從目標比率區域到其他區域的機率
approx_com=np.array(approx_com)
target_index=approx_com.tolist().index(target.tolist())
approx_com_temp=np.zeros((simulation,N))

for i in range(simulation):
    approx_com_temp[i]=approx_com[target_index]*growth[i]

#將新的價值轉換為權重
approx_com_weight_after=np.zeros((simulation,N))

#com_temp_weight的size為len(com)*simulation
for i in range(len(approx_com_temp)):
    summation=np.sum(approx_com_temp[i])
    approx_com_weight_after[i]=approx_com_temp[i]/summation

approx_after=np.zeros(len(approx_com))

for i in range(simulation):
    dis=np.sum((np.tile(approx_com_weight_after[i],(len(approx_com),1))-approx_com)**2,axis=1)
    approx_after[np.argmin(dis)]+=1

target_transition=approx_after/simulation

#%%
D=N-1
# grids_added 為想要多擺入的狀態數量
grids_added=1500

# lamda 透過 Lagrange multiplier
lamda=(np.sum(target_transition**(D/(D+1)))/(grids_added*D**(D/(D+1))))**((D+1)/D)
    
allocate_lambda=np.zeros(len(approx_com))

for i in range(len(approx_com)):
    allocate_lambda[i]=(target_transition[i]/(lamda*D))**(1/(D+1))
    
allocate_floor=np.floor(allocate_lambda)
    
for i in range(len(approx_com)):
    if allocate_floor[i]==0:
        allocate_floor[i]=1

error=(np.sqrt(D)*delta/2)*np.sum(target_transition*(1/allocate_floor))
print(error)

#%%
# 用Lagrange multiplier算出來的跟用電腦暴力解開的答案很接近
 
# def expected_error(i,n):
    
#     error=np.sqrt(D)*delta/(2*n)
    
#     return error*target_transition[i]

# def total_error(allocation):
    
#     total_error=0
    
#     for i in range(len(approx_com)):
        
#         total_error += expected_error(i,allocation[i])
    
#     return total_error

# def constraint(allocation):
    
#     return grids_added-np.sum(np.array(allocation)**D)

# allocation0 = np.full(len(approx_com),(grids_added/len(approx_com))**(1/D))

# b = (0,grids_added)

# bnds = (b,)*len(approx_com)

# con1 = {"type":"eq","fun":constraint}

# cons = [con1]

# sol = minimize(total_error,allocation0,method="SLSQP",bounds=bnds,constraints=cons,tol=1e-18)

# allocate=sol.x

#%%

# 求出每個area要擺放的狀態

def adding_grids(area):
    
    division=int(allocate_floor[area])
    tempt_weights=[]
    
    if division > 1 :
        
        if division%2 == 0:
            
            target_expand=[[approx_com[area][0]-delta*(division-1)/(division*2)],\
                           [approx_com[area][1]-delta*(division-1)/(division*2)],\
                            [approx_com[area][2]-delta*(division-1)/(division*2)]]
        
            for i in range(division-1):
                target_expand[0].append(target_expand[0][0]+(i+1)*delta/(division))
                target_expand[1].append(target_expand[1][0]+(i+1)*delta/(division))
                target_expand[2].append(target_expand[2][0]+(i+1)*delta/(division))
            
            for i in range(division):
                for j in range(division):
                    for k in range(division):
                        ww,wx,wy=target_expand[0][i],target_expand[1][j],target_expand[2][k]
                        tempt_weights.append([ww,wx,wy,1-ww-wx-wy])
        
            tempt_weights.append(approx_com[area])
        
        else:
                    
            target_expand=[[approx_com[area][0]],[approx_com[area][1]],[approx_com[area][2]]]
            
            for i in range(int((division-1)/2)):
                target_expand[0].append(target_expand[0][-1]+delta/division)
                target_expand[0].insert(0,target_expand[0][0]-delta/division)
                target_expand[1].append(target_expand[1][-1]+delta/division)
                target_expand[1].insert(0,target_expand[1][0]-delta/division)
                target_expand[2].append(target_expand[2][-1]+delta/division)
                target_expand[2].insert(0,target_expand[2][0]-delta/division)
                
            for i in range(division):
                for j in range(division):
                    for k in range(division):
                        ww,wx,wy=target_expand[0][i],target_expand[1][j],target_expand[2][k]
                        if ww==target[0] and wx==target[1] and wy==target[2]:
                            tempt_weights.append(target)
                        else:
                            tempt_weights.append([ww,wx,wy,1-ww-wx-wy])
    else:
        
        tempt_weights.append(approx_com[area])
    
    return tempt_weights

#%%
#將每個area的狀態結合成一個set

grids_of_area=[]

for area in range(len(approx_com)):
    grids_of_area.append(adding_grids(area))
    
final_grids=[]    
    
for area in range(len(approx_com)):
    for grid in range(len(grids_of_area[area])):
        final_grids.append(grids_of_area[area][grid])

for i in range(len(final_grids)):
    
    if type(final_grids[i])==np.ndarray:
        
        final_grids[i]=final_grids[i].tolist()

#%%

simulation=len(growth)
com_temp=np.zeros((simulation*len(final_grids),N)) 
com_temp_weight=np.zeros((len(com_temp),N))
com_after=np.zeros((len(final_grids),len(final_grids)))
final_transition=np.zeros((len(final_grids),len(final_grids)))
final_grids=np.array(final_grids)

for i in range(len(final_grids)):
    for j in range(simulation):
        com_temp[simulation*i+j] = final_grids[i]*growth[j]

for i in range(0,len(com_temp)):
    com_temp_weight[i]=com_temp[i]/np.sum(com_temp[i])

weight_after=np.zeros((len(final_grids),simulation,N))

for i in range(len(np.zeros((len(final_grids),simulation,N)))):
    weight_after[i]=com_temp_weight[i*simulation:(1+i)*simulation]

for i in range(len(final_grids)):
    for j in range(simulation):
        dis=np.sum((np.tile(weight_after[i][j],(len(final_grids),1))-final_grids)**2,axis=1)
        com_after[i][np.argmin(dis)]+=1

for i in range(len(final_grids)):
    final_transition[i]=com_after[i]/simulation

#%%

with open('final transition probability.txt', 'w') as outfile:
   json.dump(final_transition.tolist(), outfile)
   
with open('final weight states.txt', 'w') as outfile:
   json.dump(final_grids.tolist(), outfile)

