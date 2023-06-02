
import numpy as np
from scipy.optimize import minimize



def ll(theta, model, solver,data, pnames, out=1): # out=1 solve optimization
    """ Compute log-likelihood function """
  
    
    # Unpack and convert to numpy array
    x = np.array(data.x) 
    d = np.array(data.d)       # x is the index of the observed state (number of children)
    t = np.array(data.t)        # d is the observed decision
    dx1 = np.array(data.dx1)    # dx1 is observed change in x (birth indicator)

   
    # Update values
    model = updatepar(model, pnames, theta)
                               
    # Solve the model
    ev, pnc = solver.BackwardsInduction(model)

    # Evaluate likelihood function
    epsilon = 1e-10  # Small constant to avoid division by zero
    lik_pr = pnc[x,t]  
    #function = lik_pr * (1 - d)  + (1-lik_pr) * d 
    function = lik_pr * (1 - d) * (model.p1_list[t,1]*dx1 + model.p1_list[t,0]*(1-dx1)) + (1-lik_pr) * d *(model.p2_list[t,1]*dx1 + model.p2_list[t,0]*(1-dx1)) 

    function = np.maximum(function, epsilon)  # Add small constant to avoid zero values
    log_lik = np.log(function) #take log of likelihood function

    if out == 1:
        # Objective function (negative mean log likleihood)
        return np.mean(-log_lik)

    return model,lik_pr, pnc, ev, d,x,dx1 #, dev


def updatepar(par,parnames, parvals):
    """ Update parameters """
    for i,parname in enumerate(parnames):
        # First two parameters are scalars
        parval = parvals[i]
        setattr(par,parname,parval)
    return par


def ll3(theta, model, solver,data, pnames, out=1): # out=1 solve optimization
    """ Compute log-likelihood function """
    #global ev # Use global variable to store value function to use as starting value for next iteration
    
    # Unpack and convert to numpy array
    x = np.array(data.x)       #-1 ) # x is the index of the observed state: We subtract 1 because python starts counting at 0 
    d = np.array(data.d)
    t = np.array(data.t)        # d is the observed decision
    dx1 = np.array(data.dx1)    # dx1 is observed change in x 

    # Update values
    model = updatepar(model, pnames, theta)
                               
    # Solve the model
    ev, pnc = solver.BackwardsInduction3(model)
    

    # Evaluate likelihood functionnX
    epsilon = 1e-10  # Small constant to avoid division by zero
    lik_pr = pnc[x,t]  
    #function = lik_pr * (1 - d)  + (1-lik_pr) * d 
    function = lik_pr * (1 - d) * (model.p1_list[t,1]*dx1 + model.p1_list[t,0]*(1-dx1)) + (1-lik_pr) * d *(model.p2_list[t,1]*dx1 + model.p2_list[t,0]*(1-dx1)) 

    function = np.maximum(function, epsilon)  # Add small constant to avoid zero values
    log_lik = np.log(function)

    if out == 1:
        # Objective function (negative mean log likleihood)
        return np.mean(-log_lik)

    return model,lik_pr, pnc, ev, d,x,dx1 #, dev


def score(theta, model, solver,data, pnames): # out=1 solve optimization
    """ Compute log-likelihood function """
    #global ev # Use global variable to store value function to use as starting value for next iteration
    ev0 = solver.life1(model)
    dev_t = np.zeros((model.n,model.n))
    score_final = np.zeros((0,3))
    #log_like_sum = 0
    for t in range(model.T-2,-1, -1):
        model.p1 = model.p1_list[t]
        model.p2 = model.p2_list[t]
        model.state_transition()
        
        # Unpack and convert to numpy array
        data_t = data[(data['t']== t)]
        x = np.array(data_t.x)       #-1 ) # x is the index of the observed state: We subtract 1 because python starts counting at 0 
        d = np.array(data_t.d)       # d is the observed decision
        #dx1 = np.array(data_t.dx1)    # dx1 is observed change in x 

        # Update values
        model = updatepar(model, pnames, theta)
                                
        # Solve the model
        ev_t, pnc = model.bellman(ev0, output=2)
        ev0 = ev_t
        #dev_t += dev

        for d in range(2): # Loop over choices 
            if d == 0:
                P = model.P1
                choice_prob =  pnc
            else:
                P = model.P2
                choice_prob = 1-pnc

            dev_t += model.beta**t * choice_prob.reshape(-1, 1) * P 

        lik_pr = pnc[x]  

        F = np.eye(model.n)-dev_t # Get frechet derivative     
        N = data_t.x.size # Number of observations
        dc = model.grid 
        dc2 =  2* theta[1] * model.grid # Get derivative of cost function in utility wrt c
        pnc = pnc.reshape((model.n,1))

          # STEP 1: compute derivative of contraction operator wrt. parameters

        ## Derivative of utility function wrt. parameters
        dutil_dtheta=np.zeros((model.n, len(theta), 2)) 
        dutil_dtheta[:,0, 0] = 0 # derivative of keeping wrt RC
        dutil_dtheta[:,0, 1] = 1 # derivative of replacing wrt RC
        dutil_dtheta[:,1, 0] = dc # derivative of keeping wrt c
        dutil_dtheta[:,1, 1] = dc # derivative of replacing wrt c
        dutil_dtheta[:,2, 0] = dc2
        dutil_dtheta[:,2, 1] = dc2

        # Derivative of contraction operator wrt. utility parameters
        dbellman_dtheta=np.zeros((model.n, len(theta))) # shape is (gridsize, number of parameters)
        dbellman_dtheta[:,:] =  (pnc * dutil_dtheta[:, :, 0] + (1 - pnc) * dutil_dtheta[:, :, 1])

          # STEP 2: Compute derivative of bellman operator wrt. parameter
        dev_dtheta = np.linalg.solve(F,dbellman_dtheta)

        # STEP 3: Compute derivative of log-likelihood wrt. parameters
        score = np.zeros((N, len(theta))) # Initialize score function
        for d_loop in range(2): # Loop over decisions (keep=0, replace=1)
        # Get transition matrix
            if d_loop == 0:
                P = model.P1
            else:
                P = model.P2
            dv = dutil_dtheta[:, :, d_loop] + model.beta * P @ dev_dtheta  # derivative of choice-specific value function wrt. parameters
            choice_prob = lik_pr * (1 - d_loop) + (1-lik_pr) * d_loop # get probability of choice in loop
            score += ((d == d_loop) -  choice_prob ).reshape(-1,1) * dv[x, :] # Add derivative of log-likelihood wrt. parameters
    score_final = np.concatenate((score_final, score), axis=0)

    return score_final



def grad(theta, model, solver,data, pnames):
    """ Compute gradient of log-likelihood function """
    s = score(theta, model, solver, data,pnames)
    return -np.mean(s,0)

def hes(theta, model, solver,data, pnames):
    """ Compute Hessian of log-likelihood function as outer product of scores"""
    s= score(theta, model, solver, data, pnames)

    return s.T@s/data.shape[0]


def estimate(model,solver,data,theta0=[0.1,0.1,0.1]):
    """" Estimate model using NFXP"""
 
    
    samplesize = data.shape[0]

    # Estimate parameters
    pnames = ['eta2','eta3','mu1']
    
    # Call BHHH optimizer
    res = minimize(ll3,theta0,args = (model, solver, data, pnames), method = 'trust-ncg',jac = grad, hess = hes, tol=1e-8)
    # Update parameters
    model = updatepar(model,pnames,res.x)
    

    # Converged: "trust-ncg tends to be very conservative about convergence, and will often return status 2 even when the solution is good."
    converged   =   (res.status == 2 or res.status ==0)

    # Compute Variance-Covaiance matrix
    h = hes(res.x, model, solver,data, pnames) # Hessian
    Avar = np.linalg.inv(h*samplesize) # Variance-Covariance matrix from information matrix equality

    theta_hat = res.x # unpack estimates
    
    return model, res, pnames, theta_hat, Avar, converged