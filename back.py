# import packages used
import numpy as np
import pandas as pd
import autograd.numpy as ag_np
from autograd import grad, hessian


def BackwardsInduction(model):
    # Step 1: Initialize value and policy function
    T = model.T
    n = model.nX 

    V = np.nan + np.zeros([n, T])
    V[:, T - 1] = life(model)
    pnc =  np.nan + np.zeros([n, T])
    #dev =  np.empty(T-1, dtype=np.ndarray)
    pnc[:, T-1] = 1
    #dev0 = np.zeros((model.n,model.n))


    
    # Step 2: Backward induction

    for t in range(T - 2, -1, -1):
        # update transition matrix with birth probabilities at given age
        model.p1 = model.p1_list[t] 
        model.p2 = model.p2_list[t] 
        model.state_transition() 

        # run bellman equation to find value and choice probability for age
        ev1, pnc_t = model.bellman(ev0 = V[:, t+1], output=2)

        # store values
        V[:, t] = ev1  
        pnc[:, t] = pnc_t
        #dev[t] =  dev0 + dev1
        #dev0 = dev[t]
    return V.round(3), pnc.round(3) #, dev


def BackwardsInduction3(model):
    # Step 2: Initialize value function and policy
    T = model.T
    n = model.n #model.nX

    V = np.nan + np.zeros([n, T])
    V[:, T - 1] = life1(model)
    pnc =  np.nan + np.zeros([n, T])
    #dev =  np.empty(T-1, dtype=np.ndarray)
    pnc[:, T-1] = 1
    #dev0 = np.zeros((model.n,model.n))


    # Step 3: Backward induction

    for t in range(T - 2, -1, -1):
        model.p1 = model.p1_list[t]
        model.p2 = model.p2_list[t] 
        model.state_transition() 

        ev1, pnc_t = model.bellman(ev0 = V[:, t+1], output=2)
        V[:, t] = ev1  
        pnc[:, t] = pnc_t
        #dev[t] =  dev0 + dev1
        #dev0 = dev[t]


    return V.round(3), pnc.round(3) #, dev
    
def life1(model):
    life_value = 0
    # calculate value for infertile years a time of menopause
    for t in range(model.meno_p_years):
        life_value += (model.beta**t) * (model.eta2 * model.grid + model.eta3 * (model.grid**2))
    
    return life_value
    
def life(model):
    life_value = 0
    for t in range(model.meno_p_years):
        life_value += (model.beta**t) * (model.eta2 * model.grid[:,0] + model.eta3 * (model.grid[:,0]**2))
    
    return life_value
    
def P_list(model, data):
    #make list of birth probabilities based on birth in data
    T = model.T
    for t in range(T - 2, -1, -1):
        #subset data 
        datad0 = data[(data['d']==0) & (data['t']==t)] 
        datad1 = data[(data['d']==1) & (data['t']==t)]
        
        #Count number of observations for each dx1
        tabulate0 = datad0.dx1.value_counts() 
        tabulate1 = datad1.dx1.value_counts()
        #calculate birth probabilities
        for i in range(tabulate0.size-1):
            p1 = tabulate0[i]/sum(tabulate0) 
            p2 = tabulate1[i]/sum(tabulate1)

        # append to list
        model.p1  = np.append(p1,1-np.sum(p1))
        model.p2 = np.append(p2,1-np.sum(p2))


        model.p1_list[t] = model.p1
        model.p2_list[t] = model.p2