import numpy as np
import pandas as pd
import json
import os
import time

os.chdir(r"C:\Users\bb19x\OneDrive\桌面\何もない\Bellman Euqtion new SAA\diefferent state selection\urban planning 4 saa")

with open('final weight states.txt') as json_file:
    com = json.load(json_file)

with open('final transition probability.txt') as json_file:
    Transition = json.load(json_file)

Transition=np.array(Transition)

with open('daily returns.txt') as json_file:
    returns = json.load(json_file)

with open('expected annual return.txt') as json_file:
    mu = json.load(json_file)

with open('target.txt') as json_file:
    target = json.load(json_file)

#%%
#參數設置 
N=len(mu)
dt=1/12
returns=pd.DataFrame(returns)
cov=(returns.cov()*252).values
alpha=3
C_buy=0.001425
C_sell=0.004425
length_com=len(com)

target=com.index(target)
beta=np.exp(-0.02/12)

#policy為用來儲存每一次迭代的最佳策略
policy_PI=np.zeros(length_com)
#J為用來儲存每一次迭代total cost的結果
J=np.zeros(length_com)

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
    return (Rce_target-Rce(j))*dt


first_policy=np.zeros(length_com)
for i in range(length_com):
    first_policy[i]=target
    
policy_PI=np.vstack((policy_PI,first_policy))        
    
ite=15

def policy_evaluation(k):
    tempJ=np.zeros((ite,length_com))
    tempJ[0]=J[-2]
    for i in range(1,ite):
        for j in range(length_com):
            tempJ[i][j]=trading_cost(j,k[j])+tracking_error(k[j])+beta*np.dot(Transition[k[j]],tempJ[i-1])
    return tempJ[-1]


def policy_improvement(i,tempJ):
    compare=np.zeros(length_com)
    for k in range(length_com):
        compare[k]= trading_cost(i,k)+tracking_error(k)+beta*np.dot(Transition[k],tempJ)
    policy_PI[-1][i]=np.argmin(compare)
    return min(compare)

#%%
#迭代
tStart = time.time() 

while ~(policy_PI[-1]==policy_PI[-2]).all() :
    policy_PI=np.vstack((policy_PI,np.zeros(length_com)))
    J=np.vstack((J,np.zeros(length_com)))
    tempJ=policy_evaluation(policy_PI[-2].astype(int))
    for i in range(length_com):
        J[-1][i]=policy_improvement(i,tempJ)
    
tEnd = time.time()
print("It cost %f sec" % (tEnd - tStart))
#72 mins
#%%
#觀看收斂情況        
Dif=np.zeros((J.shape[0]-1,len(com)))

for i in range(len(com)):
    for j in range(Dif.shape[0]):
        Dif[j][i]=J[j+1][i]-J[j][i]

#查看無交易區間
        
no_trade_region=[]

for i in range(len(com)):
    if policy_PI[-1][i]==i:
        no_trade_region.append(com[i].tolist())        

#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt   

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x,y,z=[],[],[]
for i in range(len(no_trade_region)):
    x.append(no_trade_region[i][0])
    y.append(no_trade_region[i][1])
    z.append(no_trade_region[i][2])
    
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('SHCOMP')
ax.set_ylabel('SENSEX')
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('MXLA',rotation=90)
plt.subplots_adjust(left=1, right=2, top=1.5, bottom=0.6)

plt.show()

#%%

with open('approx value function (PI).txt', 'w') as outfile:
   json.dump(J.tolist(), outfile)

with open('approx convergence proof.txt', 'w') as outfile:
   json.dump(Dif.tolist(), outfile)

with open('approx policy (PI).txt', 'w') as outfile:
   json.dump(policy_PI.tolist(), outfile)

# with open('approx No Trade Region.txt', 'w') as outfile:
#    json.dump(no_trade_region, outfile)

#%%
os.chdir(r"C:\Users\bb19x\OneDrive\桌面\何もない\Bellman Euqtion new SAA\diefferent state selection\urban planning 4 saa\no trade region")

with open('No Trade Region 020_1500.txt', 'w') as outfile:
   json.dump(no_trade_region, outfile)

#%%
#with open('value function (PI).txt') as json_file:
#    J = json.load(json_file)
#J=np.array(J)
#
with open('convergence proof.txt') as json_file:
    Dif_PI = json.load(json_file)
Dif_PI=np.array(Dif_PI)
#
with open('policy (PI).txt') as json_file:
    policy_PI = json.load(json_file)
policy_PI=np.array(policy_PI)
#
#with open('No Trade Region.txt') as json_file:
#    no_trade_region = json.load(json_file)
