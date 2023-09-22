import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Read Data from cases.csv and take 7 day running average
df=pd.read_csv('../cases.csv')
data=np.array(df.iloc[:,1:670])
data=np.transpose(data)
cases_avg=np.zeros(662)

i=0
while i<662:
   cases_avg[i]=(data[i+7]-data[i])/7
   i += 1


#First Wave

#Defining Loss Function based on array p of parameters
def lf(p):

        a=1/(5.8)
        g=0.2
        b=p[0]
        cir=p[3]
        N=70000000
 
        #Implementing the SEIR Model 
        s=np.zeros(48)
        e=np.zeros(48)
        i=np.zeros(48)
        r=np.zeros(48)

        e[0]=p[1]
        i[0]=p[2]
        r[0]=0
        s[0]=N-(e[0]+i[0]+r[0])

        for t in range(47):
                s[t+1]=s[t] - ((b*i[t]*s[t])/N)
                e[t+1]=e[t] + ((b*i[t]*s[t])/N) - a*e[t]
                i[t+1]=i[t] + a*e[t] - g*i[t]
                r[t+1]=r[t] + g*i[t]

        #Finding running 7 day average of Model cases        
        sum_cases=np.cumsum(a*e/cir)
        model_avg=np.zeros(42)

        model_avg[0]=(sum_cases[6]/7)        
        i=1
        while i<42:
                model_avg[i]=(sum_cases[i+6]-sum_cases[i-1])/7
                i += 1

        #Running average from data corresponding to First wave
        data_avg=cases_avg[73:115]

        #Loss Function Expression
        l=(np.sum(np.square(np.log((data_avg))-np.log((model_avg)))))/42
        return(l)  


#Gradient Descent
p=np.array([0.3,25512,48857,25])
grad=np.array([0.,0.,0.,0.])

#Arrays for implementing perturbations
bp=np.array([0.01,0,0,0])
ep=np.array([0,100,0,0])
ip=np.array([0,0,100,0])
cirp=np.array([0,0,0,0.1])

N=70000000

#20,000 iterations of gradient descent
i=1
while (i<20000):

    #Defining the gradients with appropriate scaling
    grad[0]=(((lf(p+bp)-lf(p-bp))/0.02))/100000
    grad[1]=((lf(p+ep)-lf(p-ep))/200)*100
    grad[2]=((lf(p+ip)-lf(p-ip))/200)*100
    grad[3]=((lf(p+cirp)-lf(p-cirp))/0.2)

    norm=np.linalg.norm(grad)

    #Update of parameter array
    p=p-((1/i+1)*(grad/norm))
      
    #Defining conditions to restrict parameters within feasible bounds
    if(p[0]<=0):
        p[0]=0.01

    if(p[1]>N):  
        p[1]=N
    if(p[1]<0):
        p[1]=0 

    if(p[2]>N):  
        p[2]=N
    if(p[2]<0):
        p[2]=0 

    if(p[3]>30):
        p[3]=30
    if(p[3]<12):
        p[3]=12


    i+=1

#Final run of simulation with obtained parameters
time=np.arange(42)
a=1/(5.8)
a=1/(5.8)
g=0.2
b=p[0]
cir=p[3]
N=70000000

s=np.zeros(48)
e=np.zeros(48)
i=np.zeros(48)
r=np.zeros(48)

e[0]=p[1]
i[0]=p[2]
r[0]=0
s[0]=N-(e[0]+i[0]+r[0])

        
        
for t in range(47):
    s[t+1]=s[t] - ((b*i[t]*s[t])/N)
    e[t+1]=e[t] + ((b*i[t]*s[t])/N) - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
                

#Daily reported cases from the model
cases=a*e/cir
model_daily=cases[6:48]

#Daily reported cases from the data
data_daily=np.zeros(42) 
t=0
while(t<42):
 data_daily[t]=data[80+t]-data[79+t]
 t+=1

#Printing Output
f = open('first_wave.txt', 'w+')

strbeta ='beta : 0.407' 
stre0='E(0) : 24787'
stri0='I(0) : 48117'
strr0='R(0) : 0'
strcir='CIR : 30'

f.write(strbeta+'\n')
f.write(stre0+'\n')
f.write(stri0+'\n')
f.write(strr0+'\n')
f.write(strcir+'\n')

#Plot showing daily reported cases from the data and model
plt.plot(time,data_daily) 
plt.plot(time,model_daily)
plt.legend(['Data','Model'],loc="upper left")
plt.savefig('./first_wave_cases.jpg')
plt.close()

#Plots showing evolution of S,E,I,R values
plt.plot(time,s[6:48])
plt.savefig('./first_wave_s.jpg')
plt.close()
plt.plot(time,e[6:48])
plt.savefig('./first_wave_e.jpg')
plt.close()
plt.plot(time,i[6:48])
plt.savefig('./first_wave_i.jpg')
plt.close()
plt.plot(time,r[6:48])
plt.savefig('./first_wave_r.jpg')
plt.close()



#Second Wave

#Defining Loss Function based on array p of parameters
def lf(p):

        a=1/(5.8)
        g=0.2
        b=p[0]
        cir=p[4]
        N=70000000

        #Implementing the SEIR Model 
        s=np.zeros(48)
        e=np.zeros(48)
        i=np.zeros(48)
        r=np.zeros(48)

        e[0]=p[1]
        i[0]=p[2]
        r[0]=p[3]
        s[0]=N-(e[0]+i[0]+r[0])

        for t in range(47):
                s[t+1]=s[t] - ((b*i[t]*s[t])/N)
                e[t+1]=e[t] + ((b*i[t]*s[t])/N) - a*e[t]
                i[t+1]=i[t] + a*e[t] - g*i[t]
                r[t+1]=r[t] + g*i[t]

        #Finding running 7 day average of Model cases        
        sum_cases=np.cumsum(a*e/cir)
        model_avg=np.zeros(42)

        model_avg[0]=(sum_cases[6]/7)        
        i=1
        while i<42:
                model_avg[i]=(sum_cases[i+6]-sum_cases[i-1])/7
                i += 1

        #Running average from data corresponding to Second wave        
        data_avg=cases_avg[342:384]

        #Loss Function Expression
        l=(np.sum(np.square(np.log((data_avg))-np.log((model_avg)))))/42
        return(l)  

#Gradient Descent
p=np.array([0.55,80000,90000,18060000,25])
grad=np.array([0.,0.,0.,0.,0.])

#Arrays for implementing Perturbations
bp=np.array([0.01,0,0,0,0])
ep=np.array([0,100,0,0,0])
ip=np.array([0,0,100,0,0])
rp=np.array([0,0,0,100,0])
cirp=np.array([0,0,0,0,0.1])

N=70000000

#20,000 iterations of gradient descent
i=1
while (i<20000):

    #Defining the gradients with appropriate scaling
    grad[0]=(((lf(p+bp)-lf(p-bp))/0.02))/100000
    grad[1]=((lf(p+ep)-lf(p-ep))/200)
    grad[2]=((lf(p+ip)-lf(p-ip))/200)
    grad[3]=((lf(p+ip)-lf(p-ip))/200)
    grad[4]=((lf(p+cirp)-lf(p-cirp))/0.2)

    norm=np.linalg.norm(grad)

    #Update of parameter array
    p=p-((1/i+1)*(grad/norm))
      
    #Conditions to restrict parameters within feasible bounds
    if(p[0]<=0):
        p[0]=0.01

    if(p[1]>N):  
        p[1]=N
    if(p[1]<0):
        p[1]=0 

    if(p[2]>N):  
        p[2]=N
    if(p[2]<0):
        p[2]=0 

    if(p[3]>(0.36*N)):  
        p[3]=(0.36*N)
    if(p[3]<(0.156*N)):
        p[3]=(0.156*N)

    if(p[4]>30):
        p[4]=30
    if(p[4]<12):
        p[4]=12

    i+=1


#Final run of simulation with obtained parameters
time=np.arange(42)
a=1/(5.8)
a=1/(5.8)
g=0.2
b=p[0]
cir=p[4]
N=70000000

s=np.zeros(48)
e=np.zeros(48)
i=np.zeros(48)
r=np.zeros(48)

e[0]=p[1]
i[0]=p[2]
r[0]=p[3]
s[0]=N-(e[0]+i[0]+r[0])
       
for t in range(47):
    s[t+1]=s[t] - ((b*i[t]*s[t])/N)
    e[t+1]=e[t] + ((b*i[t]*s[t])/N) - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
                
#Daily reported cases from the model    
cases=a*e/cir
model_daily=cases[6:48]

#Daily reported cases from the data
data_daily=np.zeros(42) 
t=0
while(t<42):
 data_daily[t]=data[349+t]-data[348+t]
 t+=1

#Printing Output
f = open('second_wave.txt', 'w+')

strbeta ='beta : 0.608' 
stre0='E(0) : 79992'
stri0='I(0) : 89984'
strr0='R(0) : 18059984'
strcir='CIR : 30'

f.write(strbeta+'\n')
f.write(stre0+'\n')
f.write(stri0+'\n')
f.write(strr0+'\n')
f.write(strcir+'\n')

#Plot showing daily reported cases from the data and model
plt.plot(time,data_daily) 
plt.plot(time,model_daily)
plt.legend(['Data','Model'],loc="upper left")
plt.savefig('./second_wave_cases.jpg')
plt.close()

#Plots showing evolution of S,E,I,R values
plt.plot(time,s[6:48])
plt.savefig('./second_wave_s.jpg')
plt.close()
plt.plot(time,e[6:48])
plt.savefig('./second_wave_e.jpg')
plt.close()
plt.plot(time,i[6:48])
plt.savefig('./second_wave_i.jpg')
plt.close()
plt.plot(time,r[6:48])
plt.savefig('./second_wave_r.jpg')
plt.close()


#Third Wave

#Defining Loss Function based on array p of parameters
def lf(p):

        a=1/(5.8)
        g=0.2
        b=p[0]
        cir=p[3]
        N=70000000

        #Implementation of SEIR model
        s=np.zeros(26)
        e=np.zeros(26)
        i=np.zeros(26)
        r=np.zeros(26)

        e[0]=p[1]
        i[0]=p[2]
        r[0]=0
        s[0]=N-(e[0]+i[0]+r[0])

        for t in range(25):
                s[t+1]=s[t] - ((b*i[t]*s[t])/N)
                e[t+1]=e[t] + ((b*i[t]*s[t])/N) - a*e[t]
                i[t+1]=i[t] + a*e[t] - g*i[t]
                r[t+1]=r[t] + g*i[t]

        #Finding running 7 day average of Model cases         
        sum_cases=np.cumsum(a*e/cir)
        model_avg=np.zeros(20)

        model_avg[0]=(sum_cases[6]/7)       
        i=1
        while i<20:
                model_avg[i]=(sum_cases[i+6]-sum_cases[i-1])/7
                i += 1

        #Running average from data corresponding to Second wave
        data_avg=cases_avg[633:653]

        #Loss Function Expression
        l=(np.sum(np.square(np.log((data_avg))-np.log((model_avg)))))/20
        return(l)  


#Gradient Descent
p=np.array([1,14000,34000,30])
grad=np.array([0.,0.,0.,0.])

#Defining arrays for implementation of Perturbation
bp=np.array([0.01,0,0,0])
ep=np.array([0,100,0,0])
ip=np.array([0,0,100,0])
cirp=np.array([0,0,0,0.1])

N=70000000

#20,000 iterations of Gradient Descent
i=1
while (i<20000):

    #Defining the gradient with appropriate scaling
    grad[0]=(((lf(p+bp)-lf(p-bp))/0.02))/100000
    grad[1]=((lf(p+ep)-lf(p-ep))/200)*100
    grad[2]=((lf(p+ip)-lf(p-ip))/200)*100
    grad[3]=((lf(p+cirp)-lf(p-cirp))/0.2)/100

    norm=np.linalg.norm(grad)

    #Update of parameter array
    p=p-((1/i+1)*(grad/norm))
      
    #Conditions to restrict parameters within feasible bounds
    if(p[0]<=0):
        p[0]=0.01

    if(p[1]>N):  
        p[1]=N
    if(p[1]<0):
        p[1]=0 

    if(p[2]>N):  
        p[2]=N
    if(p[2]<0):
        p[2]=0 

    if(p[3]>30):
        p[3]=30
    if(p[3]<12):
        p[3]=12

    i+=1

#Final run of simulation with the obtained parameters  
time=np.arange(20)
a=1/(5.8)
a=1/(5.8)
g=0.2
b=p[0]
cir=p[3]
N=70000000

s=np.zeros(26)
e=np.zeros(26)
i=np.zeros(26)
r=np.zeros(26)

e[0]=p[1]
i[0]=p[2]
r[0]=0
s[0]=N-(e[0]+i[0]+r[0])
      
for t in range(25):
    s[t+1]=s[t] - ((b*i[t]*s[t])/N)
    e[t+1]=e[t] + ((b*i[t]*s[t])/N) - a*e[t]
    i[t+1]=i[t] + a*e[t] - g*i[t]
    r[t+1]=r[t] + g*i[t]
                
#Daily reported cases from the model    
cases=a*e/cir
model_daily=cases[6:26]

#Daily reported cases from the data
data_daily=np.zeros(20) 
t=0
while(t<20):
 data_daily[t]=data[640+t]-data[639+t]
 t+=1

#Printing Output
f = open('third_wave.txt', 'w+')

strbeta ='beta : 1.219' 
stre0='E(0) : 12499'
stri0='I(0) : 34696'
strr0='R(0) : 0'
strcir='CIR : 30'

f.write(strbeta+'\n')
f.write(stre0+'\n')
f.write(stri0+'\n')
f.write(strr0+'\n')
f.write(strcir+'\n')

#Plot showing daily reported cases from the data and model
plt.plot(time,data_daily) 
plt.plot(time,model_daily)
plt.legend(['Data','Model'],loc="upper left")
plt.savefig('./third_wave_cases.jpg')
plt.close()

#Plots showing evolution of S,E,I,R values
plt.plot(time,s[6:26])
plt.savefig('./third_wave_s.jpg')
plt.close() 
plt.plot(time,e[6:26])
plt.savefig('./third_wave_e.jpg')
plt.close() 
plt.plot(time,i[6:26])
plt.savefig('./third_wave_i.jpg')
plt.close() 
plt.plot(time,r[6:26])
plt.savefig('./third_wave_r.jpg')
plt.close() 

