# import packages used
import numpy as np
from types import SimpleNamespace
import scipy.optimize as optimize
par = SimpleNamespace()

def BackwardsInduction(model):
    # Step 2: Initialize value function and policy
    T = model.T
    n = model.n

    V = np.nan + np.zeros([n, T])
    V[:, T - 1] = life(model)
    pnc =  np.nan + np.zeros([n, T])
    pnc[:, T-1] = 1

    # Step 3: Backward induction

    for t in range(T - 2, -1, -1):
        ev1, pnc_t = model.bellman(ev0 = V[:, t+1], output=2)
        V[:, t] = ev1  
        pnc[:, t] = pnc_t

    return V.round(3), pnc.round(3) 
    
def life(model):
    life_value = 0
    for t in range(model.meno_p_years):
        life_value =+ (model.beta**t) * (model.eta2 * model.grid + model.eta3 * (model.grid**2))
    
    return life_value
    