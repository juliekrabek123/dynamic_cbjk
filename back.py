# import packages used
import numpy as np
import pandas as pd
import autograd.numpy as ag_np
from autograd import grad, hessian


def BackwardsInduction(model, input_data=0):
    # Step 2: Initialize value function and policy
    T = model.T
    n = model.nX #model.n

    V = np.nan + np.zeros([n, T])
    V[:, T - 1] = life(model)
    pnc =  np.nan + np.zeros([n, T])
    dev =  np.empty(T-1, dtype=np.ndarray)
    pnc[:, T-1] = 1
    p1 = np.empty(T) 
    p2 = np.empty(T) 
    dev0 = np.zeros((model.n,model.n))


    
    # Step 3: Backward induction

    for t in range(T - 2, -1, -1):
        if isinstance(input_data, pd.DataFrame):
            datad0 = input_data[(input_data['d']==0) & (input_data['t']==t)]
            datad1 = input_data[(input_data['d']==1) & (input_data['t']==t)]
            
            tabulate0 = datad0.dx1.value_counts() #Count number of observations for each dx1
            tabulate1 = datad1.dx1.value_counts()
            for i in range(tabulate0.size-1):
                #p1[t] = tabulate0[i]/sum(tabulate0) 
                #p2[t] = tabulate1[i]/sum(tabulate1)
                p1 = tabulate0[i]/sum(tabulate0) 
                p2 = tabulate1[i]/sum(tabulate1)
            #model.p1 = p1[t]
            #model.p2 = p2[t]
            model.p1  = np.append(p1,1-np.sum(p1))
            model.p2 = np.append(p2,1-np.sum(p2))
            model.state_transition()

            model.p1_list[t] = model.p1
            model.p2_list[t] = model.p2    

        ev1, pnc_t = model.bellman(ev0 = V[:, t+1], output=2)
        V[:, t] = ev1  
        pnc[:, t] = pnc_t
        dev[t] =  dev0 + dev1
        dev0 = dev[t]


    return V.round(3), pnc.round(3), dev
    
def life1(model):
    life_value = 0
    for t in range(model.meno_p_years):
        life_value =+ (model.beta**t) * (model.eta2 * model.grid + model.eta3 * (model.grid**2))
    
    return life_value
    
def life(model):
    life_value = 0
    for t in range(model.meno_p_years):
        life_value =+ (model.beta**t) * (model.eta2 * model.grid[:,0] + model.eta3 * (model.grid[:,0]**2))
    
    return life_value
    