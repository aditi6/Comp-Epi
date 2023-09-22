import numpy as np
import random
import matplotlib.pyplot as plt
import torch

# Q-1

# Likelihood function
def likelihood(s,t,y):

    sp=[0.975,0.97,0.977] # List of sensitivity values of RAT, RT-PCR and IgG tests respectively
    sen=[0.5,0.95,0.921] # List of specificity values of RAT, RT-PCR and IgG tests respectively
    p=1

    # Running an iteration of the loop for each type of test
    i=0
    while (i<3):
        if((type(y[i])==str)&(t[i]==1)): # Accounting for incompatible test suites and outcomes 
            p=0
        if((type(y[i])==int)&(t[i]==0)): # Accounting for incompatible test suites and outcomes 
            p=0  
        if (t[i]==1): # Assigning conditional probability values based on state and outcome
            if((y[i]==1)&(s[i]==1)):
                p=p*sen[i]
            elif((y[i]==0)&(s[i]==1)):
                p=p*(1-sen[i])                
            elif((y[i]==0)&(s[i]==0)):
                p=p*sp[i]
            elif((y[i]==1)&(s[i]==0)):
                p=p*(1-sp[i])               
        i=i+1   
    return p


# Q-2

states=[1,2,3,4]
state=np.random.choice(states,70000000,p=[0.1,0.32,0.01,0.57]) # Assigning states based on dissease prevalence vector given
state_list=state.tolist()                   
state_index=random.sample(state_list,16000)

# Initializing tuples s and t containing states and test suites
s=()
t=()

# Iterating for each individual
i=0
while(i<16000):

    # Randomly assigning a test suite with the given probability
    r=random.random() # Chooses a random number between 0 and 1
    if(r<=0.4):
        t=t+([0,1,1],)
    elif(r<=1):
        t=t+([0,0,1],)

    # Assigning the state vector based on the value of the state - 1,2,3 or 4
    if(state_index[i]==1):
        s=s+([1,1,0],)
    elif(state_index[i]==2):
        s=s+([0,0,1],)
    elif(state_index[i]==3):
        s=s+([1,1,1],)
    elif(state_index[i]==4):
        s=s+([0,0,0],) 
    i=i+1        


y=() # Initialising the tuple y containing test outcomes

# Iterating for each individual
i=0
while(i<16000):

    # Computing the likelihood of each of the 27 possible outcomes
    p1=likelihood(s[i],t[i],("NA","NA","NA"))
    p2=likelihood(s[i],t[i],("NA",0,"NA"))
    p3=likelihood(s[i],t[i],("NA",1,"NA"))
    p4=likelihood(s[i],t[i],(0,"NA","NA"))
    p5=likelihood(s[i],t[i],(1,"NA","NA"))
    p6=likelihood(s[i],t[i],(0,0,"NA"))
    p7=likelihood(s[i],t[i],(0,1,"NA"))
    p8=likelihood(s[i],t[i],(1,0,"NA"))
    p9=likelihood(s[i],t[i],(1,1,"NA"))
    p10=likelihood(s[i],t[i],("NA","NA",0))
    p11=likelihood(s[i],t[i],("NA",0,0))
    p12=likelihood(s[i],t[i],("NA",1,0))
    p13=likelihood(s[i],t[i],(0,"NA",0))
    p14=likelihood(s[i],t[i],(1,"NA",0))
    p15=likelihood(s[i],t[i],(0,0,0))
    p16=likelihood(s[i],t[i],(0,1,0))
    p17=likelihood(s[i],t[i],(1,0,0))
    p18=likelihood(s[i],t[i],(1,1,0))
    p19=likelihood(s[i],t[i],("NA","NA",1))
    p20=likelihood(s[i],t[i],("NA",0,1))
    p21=likelihood(s[i],t[i],("NA",1,1))
    p22=likelihood(s[i],t[i],(0,"NA",1))
    p23=likelihood(s[i],t[i],(1,"NA",1))
    p24=likelihood(s[i],t[i],(0,0,1))
    p25=likelihood(s[i],t[i],(0,1,1))
    p26=likelihood(s[i],t[i],(1,0,1))
    p27=likelihood(s[i],t[i],(1,1,1))

    p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27]
 
    # Assigning outcomes based on the probabilities found above
    r=random.random() # Chooses a random number between 0 and 1
    if(r<=p1):
        y=y+(("NA","NA","NA"),)
    elif(r<=sum(p[:2])):    
        y=y+(("NA",0,"NA"),)
    elif(r<=sum(p[:3])):
        y=y+(("NA",1,"NA"),)
    elif(r<=sum(p[:4])):
        y=y+((0,"NA","NA"),)
    elif(r<=sum(p[:5])):
        y=y+((1,"NA","NA"),) 
    elif(r<=sum(p[:6])):
        y=y+((0,0,"NA"),)                         
    elif(r<=sum(p[:7])):
        y=y+((0,1,"NA"),)
    elif(r<=sum(p[:8])):
        y=y+((1,0,"NA"),)    
    elif(r<=sum(p[:9])):
        y=y+((1,1,"NA"),)
    elif(r<=sum(p[:10])):
        y=y+(("NA","NA",0),)
    elif(r<=sum(p[:11])):
        y=y+(("NA",0,0),)
    elif(r<=sum(p[:12])):
        y=y+(("NA",1,0),)
    elif(r<=sum(p[:13])):
        y=y+((0,"NA",0),)
    elif(r<=sum(p[:14])):
        y=y+((1,"NA",0),)
    elif(r<=sum(p[:15])):
        y=y+((0,0,0),)
    elif(r<=sum(p[:16])):
        y=y+((0,1,0),)
    elif(r<=sum(p[:17])):
        y=y+((1,0,0),)
    elif(r<=sum(p[:18])):
        y=y+((1,1,0),)
    elif(r<=sum(p[:19])):
        y=y+(("NA","NA",1),)
    elif(r<=sum(p[:20])):
        y=y+(("NA",0,1),)
    elif(r<=sum(p[:21])):
        y=y+(("NA",1,1),)
    elif(r<=sum(p[:22])):
        y=y+((0,"NA",1),)
    elif(r<=sum(p[:23])):
        y=y+((1,"NA",1),)
    elif(r<=sum(p[:24])):
        y=y+((0,0,1),)
    elif(r<=sum(p[:25])):
        y=y+((0,1,1),)       
    elif(r<=sum(p[:26])):
        y=y+((1,0,1),)
    elif(r<=sum(p[:27])):
        y=y+((1,1,1),)        
    i=i+1


# Q-3

# Finding the number of occurences of each of the 27 outcomes 
state1=y.count(("NA","NA","NA"))
state2=y.count(("NA",0,"NA"))
state3=y.count(("NA",1,"NA"))
state4=y.count((0,"NA","NA"))
state5=y.count((1,"NA","NA"))
state6=y.count((0,0,"NA"))
state7=y.count((0,1,"NA"))
state8=y.count((1,0,"NA"))
state9=y.count((1,1,"NA"))
state10=y.count(("NA","NA",0))
state11=y.count(("NA",0,0))
state12=y.count(("NA",1,0))
state13=y.count((0,"NA",0))
state14=y.count((1,"NA",0))
state15=y.count((0,0,0))
state16=y.count((0,1,0))
state17=y.count((1,0,0))
state18=y.count((1,1,0))
state19=y.count(("NA","NA",1))
state20=y.count(("NA",0,1))
state21=y.count(("NA",1,1))
state22=y.count((0,"NA",1))
state23=y.count((1,"NA",1))
state24=y.count((0,0,1))
state25=y.count((0,1,1))
state26=y.count((1,0,1))
state27=y.count((1,1,1))

# Plotting the histogram
label=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27"]
freq=[state1,state2,state3,state4,state5,state6,state7,state8,state9,state10,state11,state12,state13,state14,state15,state16,state17,state18,state19,state20,state21,state22,state23,state24,state25,state26,state27]
plt.bar(label,freq,color="blue",width=0.1)
plt.savefig('./Histogram.jpg')


# Q-4

# Vectorised version of likelihood
def likelihood2(s,tuple_t,tuple_y):
    t, y = np.array(tuple_t), np.array(tuple_y) # Converting to numpy arrays
    sp = [0.975,0.97,0.977]
    sen = [0.5,0.95,0.921]

    y[y=="NA"] = -1 # Replacing the string "NA" with -1 to allow numpy manipulations
    y, t = y.astype(int), t.astype(int)
    p = np.ones(y.shape[0], dtype=float)

    for i in range(3):

        # Accounting for incompatible test suites and outcomes
        p[((y[:,i]==-1) & (t[:,i]==1)) | ((y[:,i]!=-1) & (t[:,i]==0))] = 0 

        # Assigning conditional probability values based on state and outcome
        p[(t[:,i]==1) & (y[:,i]==1) & (s[i]==1)] *= sen[i]
        p[(t[:,i]==1) & (y[:,i]==0) & (s[i]==1)] *= (1-sen[i])
        p[(t[:,i]==1) & (y[:,i]==0) & (s[i]==0)] *= sp[i]
        p[(t[:,i]==1) & (y[:,i]==1) & (s[i]==0)] *= (1-sp[i])

    return p

# Gradient Ascent using Pytorch

learning_rate = 1e-7

# Computing likelihoods using vectorized function likelihood2 
# unsqueeze(0) changes the dimension of the vector to match the tensors p1,p2 and p3 which is used in matrix multiplication later
# float() specifies the datatype of the entries as float
# torch.tensor converts them into tensors
l1, l2 = torch.tensor(likelihood2([1,1,0],t, y)).float().unsqueeze(0), torch.tensor(likelihood2([0,0,1],t, y)).float().unsqueeze(0)  
l3, l4 = torch.tensor(likelihood2([1,1,1],t, y)).float().unsqueeze(0), torch.tensor(likelihood2([0,0,0],t, y)).float().unsqueeze(0)

# Initial estimates of disease prevalence. requires_grad=True ensures that the gradients computed are remembered
p1, p2, p3 = torch.tensor([0.2], requires_grad=True), torch.tensor([0.1], requires_grad=True), torch.tensor([0.4], requires_grad=True)

# 5000 iterations of Gradient Ascent
for j in range(5000):

    ltens = p1.matmul(l1) + p2.matmul(l2) + p3.matmul(l3) + (1-p1-p2-p3).matmul(l4) #Net likelihood tensor of 16000 individuals
    lsum = torch.sum(torch.log(ltens)) # Expression of log likelihood
    lsum.backward() # Gradients computed by Backpropagation

    with torch.no_grad():
        # Values updated as per learning rate
        p1 += learning_rate * p1.grad
        p2 += learning_rate * p2.grad
        p3 += learning_rate * p3.grad
        # Ensures that the remembered values are erased so the new gradients can be conputed for the next run
        p1.grad.zero_()
        p2.grad.zero_()
        p3.grad.zero_()

print(p1)
print(p2)
print(p3)
print(1-p1-p2-p3)
