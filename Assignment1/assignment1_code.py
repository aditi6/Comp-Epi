import matplotlib.pyplot as plt
import numpy as np
s=np.zeros(500)
e=np.zeros(500)
i=np.zeros(500)
r=np.zeros(500)
time=np.arange(500)
b=0.298
a=1/(5.8)
g=0.2
s[0]=1-(100/70000000)
e[0]=(100/70000000)
for t in range(6):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    s[t+1]=s[t+1]-(100/70000000)
    e[t+1]=e[t+1]+(100/70000000)
    
for t in range(6,499,1):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    

cases = ((70000000/40)*a*e).astype(int)
greater_1000=sum(i>=1000 for i in cases)
days_to_peak=np.argmax(cases)
day_zero="29/01/2020"

print("beta_trial_and_error: " ,b)
print("num_days_to_peak: " ,days_to_peak )
print("date_of_day_0: ",day_zero)
print("num_days_greater_than_1000_cases: ",greater_1000)


plt.plot(time,cases)  
plt.xlabel("Days")
plt.ylabel("Reported Cases")
plt.show()






