import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
#%%
# original

num_states=np.array([1461,1851,2386,2912,3808,4897]) #狀態個數
cost=np.array([0.013176,0.013025,0.012903,0.012835,0.012612,0.012449]) #2000~2019間的引發的總成本
running_time=np.array([206,245,414,754,1295,2820]) # policy iteration 執行時間
error=np.array([0.012990,0.012124,0.011258,0.010392,0.009526,0.008660]) # 在狀態空間上歸類所造成的誤差(並非cost的誤差)

# delta=1.5%

num_states_15=np.array([1739,2167,2593,2947,3557]) #狀態個數
cost_15=np.array([0.012941,0.012686,0.012332,0.012436,0.012426]) #2000~2019間的引發的總成本
running_time_15=np.array([216,330,491,652,1159]) # policy iteration 執行時間
error_15=np.array([0.003720,0.002750,0.002346,0.002141,0.001913]) # 在狀態空間上歸類所造成的誤差(並非cost的誤差)
# 用real_G來分析error的收斂
# 因為真正加入的狀態數不會等於grids added
# 所以真正加入的狀態個數為總狀態個數減去原始方法狀態個數 
real_G_15=num_states_15-1461 # 原始方法delta=0.015時的狀態個數為1461

# delta=2%

num_states_20=np.array([971,1471,1725,2275,2613]) #狀態個數
cost_20=np.array([0.012914,0.013069,0.012783,0.012893,0.012843]) #2000~2019間的引發的總成本
running_time_20=np.array([62,137,203,370,502]) # policy iteration 執行時間
error_20=np.array([0.003763,0.002878,0.002572,0.002239,0.002110]) # 在狀態空間上歸類所造成的誤差(並非cost的誤差)
# 用real_G來分析error的收斂
# 因為真正加入的狀態數不會等於grids added
# 所以真正加入的狀態個數為總狀態個數減去原始方法狀態個數 
real_G_20=num_states_20-635 # 原始方法delta=0.020時的狀態個數為635

# delta=2.5%

num_states_25=np.array([780,1220,1624,1992,2548]) #狀態個數
cost_25=np.array([0.012962,0.012852,0.012154,0.012104,0.012050]) #2000~2019間的引發的總成本
running_time_25=np.array([42,100,221,288,576]) # policy iteration 執行時間
error_25=np.array([0.003790,0.003005,0.002622,0.002415,0.002189]) # 在狀態空間上歸類所造成的誤差(並非cost的誤差)
# 用real_G來分析error的收斂
# 因為真正加入的狀態數不會等於grids added
# 所以真正加入的狀態個數為總狀態個數減去原始方法狀態個數 
real_G_25=num_states_25-354 # 原始方法delta=0.025時的狀態個數為354

#%%
# time complexity

plt.xlabel("log(number of states)")
plt.ylabel("log(running time)")

plt.scatter(np.log(num_states),np.log(running_time),marker="x",label="ODP")
plt.scatter(np.log(num_states_15),np.log(running_time_15),marker="o",label="UPDP with $\Delta$=1.5%")
plt.scatter(np.log(num_states_20),np.log(running_time_20),marker="^",label="UPDP planning method with $\Delta$=2.0%")
plt.scatter(np.log(num_states_25),np.log(running_time_25),marker="s",label="UPDP planning method with $\Delta$=2.5%")
plt.legend()
plt.plot()

#%%
# 將所有的方法的資料和在一起跑迴歸
x=np.log(np.append(num_states,np.append(num_states_15,np.append(num_states_20,num_states_25))))
y=np.log(np.append(running_time,np.append(running_time_15,np.append(running_time_20,running_time_25))))
X=sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

#%%
# time complexity regression line
# 分析狀態個數與程式執行時間的關係
x=np.log(num_states)
y=np.log(running_time)
X=sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())
#%%
x=np.log(num_states_15)
y=np.log(running_time_15)
X=sm.add_constant(x)
model1 = sm.OLS(y,X)
results1 = model1.fit()
print(results1.summary())
#%%
x=np.log(num_states_20)
y=np.log(running_time_20)
X=sm.add_constant(x)
model2 = sm.OLS(y,X)
results2 = model2.fit()
print(results2.summary())
#%%
x=np.log(num_states_25)
y=np.log(running_time_25)
X=sm.add_constant(x)
model3 = sm.OLS(y,X)
results3 = model3.fit()
print(results3.summary())

#%%
# 原始方法 : Delta跟總成本的關係
plt.xlabel("$\Delta$")
plt.ylabel("total cost")
plt.xlim((0.009, 0.016))
plt.ylim((0.010, 0.016))
plt.scatter([0.01,0.011,0.012,0.013,0.014,0.015],[0.012449,0.012612,0.012835,0.012903,0.013025,0.013176])

# cost convergence regression line
x=np.array([0.01,0.011,0.012,0.013,0.014,0.015])
y=np.array([0.012449,0.012612,0.012835,0.012903,0.013025,0.013176])
X=sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())
#%%
# cost convergence
# 分析狀態個數與總成本的關係

plt.xlabel("1/(#(states)^(1/$\Theta$))")
plt.ylabel("total cost")
plt.ylim((0.010, 0.018))
plt.scatter(1/(num_states)**(1/3),cost,marker="x",label="Original state selection method")
plt.scatter(1/(num_states_15)**(1/3),cost_15,marker="o",label="Urban planning method with $\Delta$=1.5%")
plt.scatter(1/(num_states_20)**(1/3),cost_20,marker="^",label="Urban planning method with $\Delta$=2.0%")
plt.scatter(1/(num_states_25)**(1/3),cost_25,marker="s",label="Urban planning method with $\Delta$=2.5%")
plt.legend()
plt.plot()

#%%
x=1/(num_states)**(1/3)
y=cost
X=sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())
#%%
x=1/(num_states_15)**(1/3)
y=cost_15
X=sm.add_constant(x)
model1 = sm.OLS(y,X)
results1 = model1.fit()
print(results1.summary())
#%%
x=1/(num_states_20)**(1/3)
y=cost_20
X=sm.add_constant(x)
model2 = sm.OLS(y,X)
results2 = model2.fit()
print(results2.summary())
#%%
x=1/(num_states_25)**(1/3)
y=cost_25
X=sm.add_constant(x)
model3 = sm.OLS(y,X)
results3 = model3.fit()
print(results3.summary())

#%%
# error convergence

plt.xlabel("1/($real$_$G$^(1/$\Theta$))")
plt.ylabel("Total Error")

plt.scatter(1/(num_states)**(1/3),error,marker="x",label="Original state selection method")
plt.scatter(1/(real_G_15)**(1/3),error_15,marker="o",label="Urban planning method with $\Delta$=1.5%")
plt.scatter(1/(real_G_20)**(1/3),error_20,marker="^",label="Urban planning method with $\Delta$=2.0%")
plt.scatter(1/(real_G_25)**(1/3),error_25,marker="s",label="Urban planning method with $\Delta$=1.5%")
plt.legend()
plt.plot()

#%%
# error convergence regression line

x=1/(num_states)**(1/3)
y=error
X=sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())
#%%
x=1/(real_G_15)**(1/3)
y=error_15
X=sm.add_constant(x)
model1 = sm.OLS(y,X)
results1 = model1.fit()
print(results1.summary())
#%%
x=1/(real_G_20)**(1/3)
y=error_20
X=sm.add_constant(x)
model2 = sm.OLS(y,X)
results2 = model2.fit()
print(results2.summary())
#%%
x=1/(real_G_25)**(1/3)
y=error_25
X=sm.add_constant(x)
model3 = sm.OLS(y,X)
results3 = model3.fit()
print(results3.summary())