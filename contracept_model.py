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
        self.max = 5                    # Max of mileage
        # b. number of couples 
        self.N = 1000

        # b. terminal contracept age 
        self.terminal_age = 44

        # c. marriage age
        self.marriage_age = 24

        self.death_age = 76

        self.meno_p_years = self.death_age - self.terminal_age  

        # d. number of time periods doing fertile years 
        self.T = self.terminal_age - self.marriage_age

        # structual parameters
        self.p = np.array([0.9, 0.1]) 
        self.p1 = np.array([0.6, 0.4]) 
        self.p2 = np.array([0.97, 0.03])            # Transition probability
        self.eta1 = 0.13 
        self.eta2 = 0.15 
        self.eta3 = -0.05                                   # marginal utility of children
        self.mu = -0.12                                      # Cost of contraception
        self.beta = 0.99                                      # Discount factor

        # b. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 

        # c. Create grid
        self.create_grid()

    def create_grid(self):
        self.grid = np.arange(0,self.n) # milage grid
        self.cost = self.eta2*self.grid + self.eta3*(self.grid**2)   # cost function
        self.state_transition() 

    def state_transition(self):
        '''Compute transition probability matrixes conditional on choice'''
        p1 = np.append(self.p1,1-np.sum(self.p1)) # Get transition probabilities
        P1 = np.zeros((self.n,self.n)) # Initialize transition matrix
        # Loop over rows
        for i in range(self.n):
            # Check if p1 vector fits entirely
            if i <= self.n-len(p1):
                P1[i][i:i+len(p1)]=p1
            else:
                P1[i][i:] = p1[:self.n-len(p1)-i]
                P1[i][-1] = 1.0-P1[i][:-1].sum()

        # conditional on d=1, contracept
        p2 = np.append(self.p2,1-np.sum(self.p2)) # Get transition probabilities
        P2 = np.zeros((self.n,self.n)) # Initialize transition matrix
        # Loop over rows
        for i in range(self.n):
            # Check if p2 vector fits entirely
            if i <= self.n-len(p2):
                # lines where p2 vector fits entirely
                P2[i][i:i+len(p2)]=p2
            else:
                P2[i][i:] = p2[:self.n-len(p2)-i]
                P2[i][-1] = 1.0-P2[i][:-1].sum()

        self.P1 = P1
        self.P2 = P2

    def bellman(self,ev0,output=1):
        '''Evaluate Bellman operator, choice probability and Frechet derivative - written in integrated value form'''

        # Value of options:
        value_0 = self.cost + self.beta * self.P1 @ ev0 # nx1 matrix
        value_1 = self.mu + self.cost + self.beta * self.P2 @ ev0   # nx1 matrix

        # recenter Bellman by subtracting max(VK, VR)
        maxV = np.maximum(value_0, value_1) 
        # d = np.zeros(self.n)
        # for i in range(self.n):
        #     if maxV[i] == value_0[i]:
        #         d[i] = 0
        #     else:
        #         d[i] = 1

        logsum = (maxV + np.log(np.exp(value_0-maxV)  +  np.exp(value_1-maxV)))  # Compute logsum to handle expectation over unobserved states
        ev1 = logsum # Bellman operator as integrated value

        if output == 1:
            return ev1

        # Compute choice probability of not contracepting
        pnc = 1/(1+np.exp(value_1-value_0))       
        
        if output == 2:
            return ev1, pnc
        
        
        if output == 4:
            return ev1, pnc, d, value_0, value_1

        # Compute derivative of Bellman operator
        dev1 = self.dbellman(pnc)

        return ev1, pnc, dev1

    def dbellman(self,pnc): 
        '''Compute derivative of Bellman operator'''
        dev1 = np.zeros((self.n,self.n))
        for d in range(2): # Loop over choices 
            if d == 0:
                P = self.P1
                choice_prob =  pnc
            else:
                P = self.P2
                choice_prob = 1-pnc

            dev1 += self.beta * choice_prob.reshape(-1, 1) * P 
        
        return dev1



    def sim_data(self, pnc):
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
            d[it,:] = u_d[it,:] < 1-pnc[x[it,:]]   # Contracept = 1 , not contracept = 0 for s in range(T*N):


        for s in range(T):
            for r in range(N):
                d[s,r] = u_d[s,r] < 1-pnc[x[s,r]]   # Contracept = 1 , not contracept = 0 for s in range(T*N):
                if d[s,r] == 0:
                # Find states and choices
                    csum_p1 = np.cumsum(self.p1)               # Cumulated sum of p 
                ## this loop will iterate twice and dx1 will be incremented by either 0 or 1 on each iteration depending on the result of the comparison
                    dx1[s,r] = 0
                    for val in csum_p1:
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



    def read_data(self): 
        data = np.loadtxt(open("carro-mira.csv"), delimiter=",")
        idx = data[:,3]             # couple id
        t = data[:,4]             # year
        cc = data[:,10]         # contraception choice
        d = data[:,14]              # decision
        x = data[:,9]               # number of children
        dx1 = data[:,8]            # birth indicator


     
        # change type to integrer
        x = x.astype(int)
        dx1 = dx1.astype(int)

        # Collect in a dataframe
        
        data = {'id': idx, 'year' : t,'contraception choice':cc, 'd': d, 'x': x, 'dx1': dx1}
        df= pd.DataFrame(data) 

        # Remove observations with missing lagged mileage
        df = df.drop(df[df['contraception choice'] == 3].index, axis=0)
        df = df.drop(df[df['x'] > 4].index, axis=0)



        # save data
        dta = df.drop(['contraception choice'],axis=1)
        
        return dta
    
    def life(self):
        life_value = 0
        for t in range(self.meno_p_years):
            life_value =+ (self.beta**t) * (self.eta2 * self.grid + self.eta3 * (self.grid**2))
        
        return life_value
        