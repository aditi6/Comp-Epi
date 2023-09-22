import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Simulation of basecase scenario to align the timeline to the peak mentioned in the assignment

#SEIR Dynamics
s=np.zeros(500)
e=np.zeros(500)
i=np.zeros(500)
r=np.zeros(500)
time=np.arange(500)

#Beta and cir values were taken from the result in assignment 2 and multiplied by 2.6 as instructed
b=2.6*0.608
a=1/(5.8)
g=0.2
cir=30
N=12400000 

s[0]=0.6-(100/N)
e[0]=(100/N)
i[0]=0
r[0]=0.4 #1-susceptible fraction

#Seeding infection for the first 6 days
for t in range(6):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    s[t+1]=s[t+1]-(100/N)
    e[t+1]=e[t+1]+(100/N)
    
for t in range(6,499,1):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    


cases = ((N/cir)*a*e).astype(int)
days_to_peak=np.argmax(cases)
#The number of days to peak was found to be 53. Based on this, Day0 was 29/11/2021. The simulation began from there and lasted till 31/03/22 - 123 days


#Baseline case

#SEIR Dynamics
s=np.zeros(123)
e=np.zeros(123)
i=np.zeros(123)
r=np.zeros(123)

time=np.arange(123)
cases=np.zeros(123)
beds=np.zeros(123)
icu=np.zeros(123)

b=2.6*0.608
a=1/(5.8)
g=0.2
cir=30
N=12400000

s[0]=0.6-(100/N)
e[0]=(100/N)
i[0]=0
r[0]=0.4 #1-susceptible fraction

#Seeding infection for the first 6 days
for t in range(6):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    s[t+1]=s[t+1]-(100/N)
    e[t+1]=e[t+1]+(100/N)
    
for t in range(6,122,1):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    
cases = ((N/cir)*a*e).astype(int)

beds_perday=(cases*0.02).astype(int) #daily addition to the number of beds required

#To account for the 10 day occupancy, a windowed sum (window of 10) is taken

beds_tot=np.cumsum(beds_perday)
i=0
while(i<10):
     beds[i]=beds_tot[i]
     i+=1
i=10
while(i<123):
     beds[i]=beds_tot[i]-beds_tot[i-10]
     i+=1

icu_perday=(cases*0.02*0.05).astype(int) #daily addition to the number of icus required
icu_tot=np.cumsum(icu_perday)

#To account for the 10 day occupancy, a windowed sum (window of 10) is taken
i=0
while(i<10):
     icu[i]=icu_tot[i]
     i+=1
i=10
while(i<123):
     icu[i]=icu_tot[i]-icu_tot[i-10]
     i+=1 

jan_avg_base=np.average(cases[33:64])
feb_avg_base=np.average(cases[64:92])
mar_avg_base=np.average(cases[92:123])


days_to_peak=np.argmax(cases)
#Days to peak was found to be 53, and the date of the peak was 21 January 2022, as per our alignment
peak_cases_base=np.max(cases)
peak_beds_base=np.max(beds)
peak_icu_base=np.max(icu)

#saved in order to generate a plot later
cases_base=cases 
beds_base=beds
icu_base=icu


#Curfew case

#SEIR Dynamics
s=np.zeros(123)
e=np.zeros(123)
i=np.zeros(123)
r=np.zeros(123)

time=np.arange(123)
cases=np.zeros(123)
beds=np.zeros(123)
icu=np.zeros(123)

b=2.6*0.608
a=1/(5.8)
g=0.2
cir=30
N=12400000

s[0]=0.6-(100/N)
e[0]=(100/N)
i[0]=0
r[0]=0.4 #1-susceptible fraction

#Seeding the infection
for t in range(6):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    s[t+1]=s[t+1]-(100/N)
    e[t+1]=e[t+1]+(100/N)

#Before the curfew started on 1/1/22    
for t in range(6,32,1):
    s[t+1]=s[t] - b*i[t]*s[t]
    e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
    

for x in range(12):

    #Saturday and Sunday
    b= 0.55*2.6*0.608
    for t in range(32+(7*x),32+(7*x)+2,1):
        s[t+1]=s[t] - b*i[t]*s[t]
        e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
        i[t+1]=i[t] + a*e[t] - g*i[t]
        r[t+1]=r[t] + g*i[t]
 
    #Monday to Friday
    b= 0.9*2.6*0.608 
    for t in range(34+(7*x),34+(7*x)+5,1):
        s[t+1]=s[t] - b*i[t]*s[t]
        e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
        i[t+1]=i[t] + a*e[t] - g*i[t]
        r[t+1]=r[t] + g*i[t]

#Last 6 days
x=12

#Saturday and Sunday
b= 0.55*2.6*0.608
for t in range(32+(7*x),32+(7*x)+2,1):
        s[t+1]=s[t] - b*i[t]*s[t]
        e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
        i[t+1]=i[t] + a*e[t] - g*i[t]
        r[t+1]=r[t] + g*i[t]

#Monday to Thursday 31/03/22
b= 0.9*2.6*0.608 
for t in range(34+(7*x),34+(7*x)+4,1):
        s[t+1]=s[t] - b*i[t]*s[t]
        e[t+1]=e[t] + b*i[t]*s[t] - a*e[t]
        i[t+1]=i[t] + a*e[t] - g*i[t]
        r[t+1]=r[t] + g*i[t]      
        
cases = ((N/cir)*a*e).astype(int)

beds_perday=(cases*0.02).astype(int) #daily addition to the number of beds required

#To account for the 10 day occupancy, a windowed sum (window of 10) is taken
beds_tot=np.cumsum(beds_perday)
i=0
while(i<10):
     beds[i]=beds_tot[i]
     i+=1
i=10
while(i<123):
     beds[i]=beds_tot[i]-beds_tot[i-10]
     i+=1

icu_perday=(cases*0.02*0.05).astype(int) #daily addition to the number of icus required

#To account for the 10 day occupancy, a windowed sum (window of 10) is taken
icu_tot=np.cumsum(icu_perday)
i=0
while(i<10):
     icu[i]=icu_tot[i]
     i+=1
i=10
while(i<123):
     icu[i]=icu_tot[i]-icu_tot[i-10]
     i+=1   


jan_avg_curfew=np.average(cases[33:64])
feb_avg_curfew=np.average(cases[64:92])
mar_avg_curfew=np.average(cases[92:123])


days_to_peak=np.argmax(cases)
#Days to peak was found to be 58, thus the date of the peak was 26 January 2022
peak_cases_curfew=np.max(cases)
peak_beds_curfew=np.max(beds)
peak_icu_curfew=np.max(icu)

 #saved in order to generate a plot later
cases_curfew=cases
beds_curfew=beds 
icu_curfew=icu



#Printing Output
f = open('output.txt', 'w+')
Scenario=[ 'Base case', 'Curfew case', 'Reduction']
Jan_cases=[jan_avg_base , jan_avg_curfew , (jan_avg_base-jan_avg_curfew)]
Feb_cases=[feb_avg_base , feb_avg_curfew , (feb_avg_base-feb_avg_curfew)]
March_cases=[mar_avg_base , mar_avg_curfew , (mar_avg_base-mar_avg_curfew)]

peak_time=['21/11/22' , '26/11/22', '5 days']
peak_cases=[peak_cases_base, peak_cases_curfew , (peak_cases_base - peak_cases_curfew)]
peak_beds=[peak_beds_base, peak_beds_curfew , (peak_beds_base - peak_beds_curfew)]
peak_icu=[peak_icu_base, peak_icu_curfew, (peak_icu_base - peak_icu_curfew) ]

df1 = pd.DataFrame({'Scenario':Scenario,'January cases average':Jan_cases, 'February cases average':Feb_cases, 'March cases average':March_cases})
df2 = pd.DataFrame({'Scenario':Scenario, 'Peak date':peak_time, 'Peak cases':peak_cases, 'Peak Hospital beds':peak_beds, 'Peak ICU ventilators':peak_icu})

table1=df1.to_string(index=False)
table2=df2.to_string(index=False)

f.write('Table 1\n')
f.write(table1+'\n')
f.write('\n')
f.write('Table 2\n')
f.write(table2)

#Plot showing daily reported cases from the data and model
plt.plot(time,cases_base) 
plt.plot(time,cases_curfew)
plt.legend(['Base','Curfew'],loc="upper left")
plt.savefig('./cases.jpg')
plt.close()

plt.plot(time,beds_base) 
plt.plot(time,beds_curfew)
plt.legend(['Base','Curfew'],loc="upper left")
plt.savefig('./hospital_beds.jpg')
plt.close()

plt.plot(time,icu_base) 
plt.plot(time,icu_curfew)
plt.legend(['Base','Curfew'],loc="upper left")
plt.savefig('./ICU_beds.jpg')
plt.close()






