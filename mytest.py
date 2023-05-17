# Zurcher class: Contains model parts for Rust's engine repplacement model Rust(Ecta, 1987)

#import packages
import numpy as np
import time
import pandas as pd

class child_model():
    def __init__(self,**kwargs):
        self.setup(**kwargs)

    def setup(self,**kwargs):     
  
        # a) parameters
        # Spaces
        self.n = 5                     # Number of grid points
        self.max = 5    
                
        # b. number of couples 
        self.N = 1000

        # b. terminal contracept age 
        self.terminal_age = 44

        # c. marriage age
        self.marriage_age = 34

        # d. number of time periods doing fertile years 
        self.T = self.terminal_age - self.marriage_age
                # Max of children

        # structual parameters
        self.p = np.array([0.3, 0.7]) 
        self.p2 = np.array([0.97, 0.03])            # Transition probability
        self.eta1 = 0.13 
        self.eta2 = 0.15 
        self.eta3 = -0.5                                   # marginal utility of children
        self.mu = -0.12                                      # Cost of contraception
        self.beta = 0.9999                                   # Discount factor
        #self.utility = eta2*x + eta3*x**2 + mu*d                         

        # b. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 

        # c. Create grid
        self.create_grid()

    def create_grid(self):
        self.grid = np.arange(0,self.n) # grid
        self.cost = self.eta2*self.grid + self.eta3*(self.grid**2)  # cost function
        #self.dc = self.grid*0.001 - 2*self.eta2*(self.grid*0.001)
        self.state_transition() 

    def state_transition(self):
        '''Compute transition probability matrixes conditional on choice'''
        p = np.append(self.p,1-np.sum(self.p)) # Get transition probabilities
        P1 = np.zeros((self.n,self.n)) # Initialize transition matrix
        # Loop over rows
        for i in range(self.n):
            # Check if p vector fits entirely
            if i <= self.n-len(p):
                # lines where p vector fits entirely
                P1[i][i:i+len(p)]=p
            else:
                P1[i][i:] = p[:self.n-len(p)-i]
                P1[i][-1] = 1.0-P1[i][:-1].sum()
                
        p2 = np.append(self.p2,1-np.sum(self.p2)) # Get transition probabilities
        P2 = np.zeros((self.n,self.n)) # Initialize transition matrix
        # Loop over rows
        for i in range(self.n):
            # Check if p vector fits entirely
            if i <= self.n-len(p2):
                # lines where p vector fits entirely
                P2[i][i:i+len(p2)]=p2
            else:
                P2[i][i:] = p2[:self.n-len(p2)-i]
                P2[i][-1] = 1.0-P2[i][:-1].sum()

        # conditional on d=1, replacement
        #P2 = np.zeros((self.n,self.n))
        # Loop over rows
        #for i in range(self.n):
        #    P2[i][:len(p)]=p
        #P2 = np.identity(self.n)
        self.P1 = P1
        self.P2 = P2

    

    def bellman(self,ev0=np.zeros(1),output=1):

        # Value of options:

        value_0 = self.cost  + self.beta * ev0 # nx1 matrix
        value_1 = self.cost + self.mu + self.beta * ev0   # nx1

        
        # recenter Bellman by subtracting max(VK, VR)
        maxV = np.maximum(value_0,value_1) 
        logsum = (maxV + np.log(np.exp(value_0-maxV)  +  np.exp(value_1-maxV)))  # This is the Logsum 
        ev1 = self.P1@logsum

        if output == 1:
            return ev1

        # Compute choice probability
        pk = 1/(1+np.exp(value_1-value_0))
        #pk= np.divide(np.exp(value_0),(np.exp(value_1)+np.exp(value_0)))
        
        if output == 2:
            return ev1, pk

        # Compute Frechet derivative
        dev1 =self.dbellman(pk)
        
        if output == 3:
            return ev1, pk, dev1


    def dbellman(self,pk): 
        '''Compute derivative of Bellman operator'''
        dev1 = np.zeros((self.n,self.n))
        for d in range(2): # Loop over choices 
            if d == 0:
                P = self.P1
                choice_prob =  pk
            else:
                P = self.P2
                choice_prob = 1-pk

            dev1 += self.beta * choice_prob.reshape(-1, 1) * P 
        
        return dev1

    # def read_busdata(self, bustypes = [1,2,3,4]): 
    #     data = np.loadtxt(open("busdata1234.csv"), delimiter=",")
    #     idx = data[:,0]             # bus id
    #     bustype = data[:,1]         # bus type
    #     dl = data[:,4]              # laggend replacement dummy
    #     d = np.append(dl[1:], 0)    # replacement dummy
    #     x = data[:,6]               # Odometer

    #     # Discretize odometer data into 1,2,...,n
    #     x = np.ceil(x*self.n/(self.max*1000))

    #     # Montly mileage
    #     dx1 = x-np.append(0,x[0:-1])
    #     dx1 = dx1*(1-dl)+x*dl
    #     dx1 = np.where(dx1>len(self.p),len(self.p),dx1) # We limit the number of steps in mileage

    #     # change type to integrer
    #     x = x.astype(int)
    #     dx1 = dx1.astype(int)

    #     # Collect in a dataframe
    #     remove_first_row_index=idx-np.append(0,idx[:-1])
    #     data = {'id': idx,'bustype':bustype, 'd': d, 'x': x, 'dx1': dx1, 'boolean': remove_first_row_index}
    #     df= pd.DataFrame(data) 

    #     # Remove observations with missing lagged mileage
    #     df = df.drop(df[df.boolean!=0].index)

    #     # Select bustypes 
    #     for j in [1,2,3,4]:
    #         if j not in bustypes:
    #             df = df.drop(df[df.bustype==j].index) 

    #     # save data
    #     dta = df.drop(['id','bustype','boolean'],axis=1)
        
    #     return dta

    # def sim_data(self,N,T,pk): 

    #     # Index 
    #     idx = np.tile(np.arange(1,N+1),(T,1))  
    #     t = np.tile(np.arange(1,T+1),(N,1)).T
            
    #     # Draw random numbers
    #     #u_init = np.random.randint(1,size=(1,N)) # initial condition
    #     u_dx = np.random.rand(T,N) # mileage
    #     u_d = np.random.rand(T,N) # choice
            
    #     # Find states and choices

    #     x = np.zeros((T,N),dtype=int)
    #     x1 =  np.zeros((T,N),dtype=int)
    #     d = np.nan + np.zeros((T,N))
    #     x[0,:] = np.zeros((1,N)) #u_init # initial condition
    #     for it in range(T):
    #         d[it,:] = u_d[it,:]<pk[x[it,:]]  # contraception = 1 ,no contraception = 0    

    #     u_dxd = u_dx*d


    #     csum_p = np.cumsum(self.p)
    #     dx1 = 0
    #     for val in csum_p:
    #         dx1 += u_dxd>val

    #     for it in range(T):
    #         x1[it,:] = np.minimum((x[it,:] + dx1[it,:] ), self.n-2) # State transition, minimum to avoid exceeding the maximum mileage
    #         if it < T-1:
    #             x[it+1,:] = x1[it,:]
            
            
    #     # reshape 
    #     idx =  np.reshape(idx,T*N,order='F')
    #     t = np.reshape(t,T*N,order='F')
    #     d = np.reshape(d,T*N,order='F')
    #     x = np.reshape(x,T*N,order='F') + 1 # add 1 to make index start at 1 as in data - 1,2,...,n
    #     x1 = np.reshape(x1,T*N,order='F') + 1 # add 1 to make index start at 1 as in data - 1,2,...,n
    #     dx1 = np.reshape(dx1,T*N,order='F')


    #     data = {'id': idx,'t': t, 'd': d, 'x': x, 'dx1': dx1, 'x1': x1}
    #     df= pd.DataFrame(data) 

    #     return df

    def sim_data(self, pk):
        # Set N (number of couples) and T (fertile years)
        N = self.N
        T = self.T

        # Set random seed 
        np.random.seed(2020)

        # Index 
        idx = np.tile(np.arange(1,N+1),(T,1))  
        t = np.tile(np.arange(1,T+1),(N,1)).T
            
        # Draw random numbers
        # u_init =np.nan + np.zeros((1,N)) # initial condition
        u_d = np.random.rand(T,N) 
        u_dx = np.random.rand(T,N)                # decision/choice

        # Find states and choices
        #u_dx  = np.zeros((T,N), dtype=int)
        ## state 
        x  = np.zeros((T,N), dtype=int)
        ## state next period 
        x1 = np.zeros((T,N), dtype=int)

        dx1 = np.zeros((T,N), dtype=int)
        ## decision/choices
        d  = np.nan + np.zeros((T,N))
        ## initial condition
        x[0,:] = np.zeros((1,N)) # u_init.astype(int)

        for it in range(T):
            d[it,:] = u_d[it,:] < 1-pk[x[it,:]]   # Contracept = 1 , not contracept = 0 for s in range(T*N):


        for s in range(T):
            for r in range(N):
                d[s,r] = u_d[s,r] < 1-pk[x[s,r]]   # Contracept = 1 , not contracept = 0 for s in range(T*N):
                if d[s,r] == 0:
                # Find states and choices
                    csum_p = np.cumsum(self.p)               # Cumulated sum of p 
                ## this loop will iterate twice and dx1 will be incremented by either 0 or 1 on each iteration depending on the result of the comparison
                    dx1[s,r] = 0
                    for val in csum_p:
                        # if (u_dx > val).all():
                        #     dx1[it,:] += 1
                        dx1[s,r] += u_dx[s,r]>val

                else:
                    # Find states and choices
                    csum_p2 = np.cumsum(self.p2)               # Cumulated sum of p 
                ## this loop will iterate twice and dx1 will be incremented by either 0 or 1 on each iteration depending on the result of the comparison
                    dx1[s,r] = 0

                    for val in csum_p2:
                        dx1[s,r] += u_dx[s,r]>val
                # Find states and choices
            
                x1[s,r] = np.minimum(x[s,r]+dx1[s,r], self.n-1) # State transition, minimum to avoid exceeding the maximum number of children
            
            # Ensure that the number of children cannot decrease
            #x1[s,r] = np.maximum(x1[s,r], x[s,r])

                if s < T-1:
                    x[s+1,r] = x1[s,r]


        # reshape 
        idx = np.reshape(idx,T*N,order='F')
        t   = np.reshape(t,T*N,order='F')
        d   = np.reshape(d,T*N,order='F')
        x   = np.reshape(x,T*N,order='F')   # add 1 to make index start at 1 as in data - 1,2,...,n
        x1  = np.reshape(x1,T*N,order='F')  # add 1 to make index start at 1 as in data - 1,2,...,n
        dx1 = np.reshape(dx1,T*N,order='F')

        data = {'id': idx,'t': t, 'd': d, 'x': x, 'dx1': dx1, 'x1': x1}
        df = pd.DataFrame(data) 

        return(df)


    
    # def sim_data(self,pk): 
        
    #     """ Simulate data """

    #     # Set N (number of couples) and T (fertile years)
    #     N = self.N
    #     T = self.T

    #     # Set random seed 
    #     np.random.seed(2020)

    #     # Index 
    #     idx = np.tile(np.arange(1,N+1),(T,1))  
    #     t = np.tile(np.arange(1,T+1),(N,1)).T
            
    #     # Draw random numbers
    #    # u_init =np.nan + np.zeros((1,N)) # initial condition
    #     u_dx = np.random.rand(T,N)               # birth 
    #     u_d = np.random.rand(T,N)                # decision/choice
        
    #     # Find states and choices
    #     csum_p = np.cumsum(self.p)               # Cumulated sum of p 
    #     ## this loop will iterate twice and dx1 will be incremented by either 0 or 1 on each iteration depending on the result of the comparison
    #     dx1 = 0
    #     for val in csum_p:
    #         dx1 += u_dx>val

    #     # Find states and choices
    #     ## state 
    #     x  = np.zeros((T,N), dtype=int)
    #     ## state next period 
    #     x1 = np.zeros((T,N), dtype=int)
    #     ## decision/choices
    #     d  = np.nan + np.zeros((T,N))
    #     ## initial condition
    #     x[0,:] = np.zeros((1,N)) # u_init.astype(int)
    #     ## Overall, this code simulates a stochastic process that models the decision to use contraception based on the current number of children (x) and a random draw (u_d), 
    #     ## and updates the number of children based on the decision to use contraception and a second random draw (dx1). 
    #     ## The pk variable is used to parameterize the probability of using contraception based on the current number of children.
    #     for it in range(T):
    #         d[it,:] = u_d[it,:] < 1-pk[x[it,:]]   # Contracept = 1 , not contracept = 0   
    #         #if not any(d[it,:]): # if not contracept
    #         #x1[it,:] = np.minimum((x[it,:]+1) + dx1[it,:] , self.n-1) # State transition, minimum to avoid exceeding the maximum number of children
    #         #else: # if contracept
    #         x1[it,:] = np.minimum(x[it,:]*(1-d[it,:]) + dx1[it,:] , self.n-1) # State transition, minimum to avoid exceeding the maximum number of children
            
    #         # Ensure that the number of children cannot decrease
    #         #x1[it,:] = np.maximum(x1[it,:], x[it,:])

    #         #if it > 0 and np.all(x[it-1,:] == self.n-1):
    #         #    d[it,:] = 1

    #         if it < T-1:
    #             x[it+1,:] = x1[it,:]

       
    #     # reshape 
    #     idx = np.reshape(idx,T*N,order='F')
    #     t   = np.reshape(t,T*N,order='F')
    #     d   = np.reshape(d,T*N,order='F')
    #     x   = np.reshape(x,T*N,order='F')   # add 1 to make index start at 1 as in data - 1,2,...,n
    #     x1  = np.reshape(x1,T*N,order='F')  # add 1 to make index start at 1 as in data - 1,2,...,n
    #     dx1 = np.reshape(dx1,T*N,order='F')

    #     data = {'id': idx,'t': t, 'd': d, 'x': x, 'dx1': dx1, 'x1': x1}
    #     df = pd.DataFrame(data) 

    #     return df

    # def eqb(self, pk):
    #     # Inputs
    #     # pk: choice probability

    #     # Outputs    
    #     # pp: Pr{x} (Equilibrium distribution of mileage)
    #     # pp_K: Pr{x,i=Keep}
    #     # pp_R: Pr{x,i=Replace}
    #     tmp =self.P1[:,1:self.n] * pk[1:self.n]
    #     pl = np.hstack(((1-np.sum(tmp,1,keepdims=True)), tmp)) 

    #     pp = self.ergodic(pl)

    #     pp_K = pp.copy()    
    #     pp_K[0] = self.p[0]*pp[0]*pk[0]
    #     pp_R = (1-pk)*pp # Vær opmærksom på at dette måske skal ændres hvis sandsynligheden for uønsket børn ændre sig

    #     return pp, pp_K, pp_R

    # def ergodic(self,p):
    #     #ergodic.m: finds the invariant distribution for an NxN Markov transition probability: q = qH , you can also use Succesive approximation
    #     n = p.shape[0]
    #     if n != p.shape[1]:
    #         print('Error: p must be a square matrix')
    #         ed = np.nan
    #     else:
    #         ap = np.identity(n)-p.T
    #         ap = np.concatenate((ap, np.ones((1,n))))
    #         ap = np.concatenate((ap, np.ones((n+1,1))),axis=1)

    #         # find the number of linearly independent columns
    #         temp, _ = np.linalg.eig(ap)
    #         temp = ap[temp==0,:]
    #         rank = temp.shape[1]
    #         if rank < n+1:
    #             print('Error: transition matrix p is not ergodic')
    #             ed = np.nan
    #         else:
    #             ed = np.ones((n+1,1))
    #             ed[n] *=2
    #             ed = np.linalg.inv(ap)@ed
    #             ed = ed[:-1]
    #             ed = np.ravel(ed)

    #     return ed
