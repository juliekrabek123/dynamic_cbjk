
import numpy as np
import scipy.optimize as optimize



def estimate(model,solver,data,theta0=[0,0,0],twostep=0):
    """" Estimate model using NFXP"""
    global ev
    ev = np.zeros((model.n)) 
    
    samplesize = data.shape[0]
    
    # STEP 1: Find p non-parametrically
    datad0 = data[data['d']==0]
    datad1 = data[data['d']==1]
    
    tabulate0 = datad0.dx1.value_counts() #Count number of observations for each dx1
    tabulate1 = datad1.dx1.value_counts()
    p = [tabulate0[i]/sum(tabulate0) for i in range(tabulate0.size-1)]
    p2 = [tabulate1[i]/sum(tabulate1) for i in range(tabulate1.size-1)]

    # STEP 2: Estimate structual parameters
    model.p[:] = p # Use first step estimates as starting values for p
    model.p2[:] = p2
    # Estimate mu and eta2
    pnames = ['mu','eta2','eta3']
    
    # Call optimizer
    res = optimize.minimize(ll,theta0,args = (model, solver, data, pnames), method = 'trust-ncg', tol=1e-8)
    # Update parameters
    model = updatepar(model,pnames,res.x)

    # Converged: "trust-ncg tends to be very conservative about convergence, and will often return status 2 even when the solution is good."
    converged   =   (res.status == 2 or res.status ==0) 

    # Compute Variance-Covaiance matrix
    #h = hes(res.x, model, solver,data, pnames) # Hessian
    #Avar = np.linalg.inv(h*samplesize) # Variance-Covariance matrix from information matrix equality

    theta_hat = res.x # unpack estimates
    
    return model, res, pnames, theta_hat, converged



def ll(theta, model, solver,data, pnames, out=1, no_guess = False): # out=1 solve optimization
    """ Compute log-likelihood function """
    global ev # Use global variable to store value function to use as starting value for next iteration
    
    # Unpack and convert to numpy array
    x = np.array(data.x)       #-1 ) # x is the index of the observed state: We subtract 1 because python starts counting at 0 
    d = np.array(data.d)        # d is the observed decision
    dx1 = np.array(data.dx1)    # dx1 is observed change in x 

    if no_guess == True:        # Set ev to zero instead of using global variable
        ev = np.zeros((model.n))
    
    # Update values
    model = updatepar(model, pnames, theta)
    model.p = np.abs(model.p)               # helps BHHH which is run as unconstrained optimization
    model.create_grid()                     # Update grid
    ev0 = ev                                # Use previous value function as starting value

    # Solve the model
    ev, pnc, dev = solver.poly(model.bellman, V0=ev0 ,beta=model.beta, output=3)

    # Evaluate likelihood function
    lik_pr = pnc[x,t]                                  # Get probability of not contracepting given observed state    
    function =  lik_pr * (1 - d) * (model.p1[1]*dx1 + model.p1[0]*(1-dx1)) + (1-lik_pr) * d *(model.p2[1]*dx1 + model.p2[0]*(1-dx1)) # get probability of making observed choice
    log_lik = np.log(function)                   # Compute log-likelihood-contributions
    
    # add on log like for birth porbabilities
    if theta.size>3: # theta > 2 if there are parameters for p
        p = np.append(model.p,1-np.sum(model.p))    # Add residual probability to p
        if any(p<=0):
            log_lik -= 100000*p[dx1]                # Penalize if p is negative
        else:
            log_lik += np.log(p[dx1])               # Add log-likelihood contribution
        
    else:
        p = np.nan

    if out == 1:
        # Objective function (negative mean log likleihood)
        return np.mean(-log_lik)

    return model,lik_pr, pk, ev, dev, d,x,dx1