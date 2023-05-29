# import packages used
import numpy as np
from types import SimpleNamespace
import scipy.optimize as optimize
par = SimpleNamespace()

par.mu = -0.12
par.eta2 = 0.15
par.eta3 = -0.3
par.m_max =5
par.m_min = 0
par.D = 2
par.T = 25

def util(N,d,par):
    return par.eta2*m + par.eta3*(m**2) + par.mu*d


def solve_last_period(par):

    # a. allocate
    c_grid = np.linspace(par.m_min,par.m_max,par.m_max)
    v_func = np.empty(par.m_max)
    d = (0,1)
    d_func = np.empty(par.m_max)

    # b. solve
    for i,m in enumerate(c_grid):
        for j,d in range d_func:

        # i. objective
        obj = lambda d: -util(d,m,par)

        # ii. optimizer
        x0 = m/2 # initial value
        result = optimize.minimize(obj,[d],method='L-BFGS-B',bounds=((1e-8),))

        # iii. save
        v_func[i] = -result.fun
        d_func[j] = result.x
        
    return m_grid,v_func,d_func