from types import SimpleNamespace
import numpy as np
import pandas as pd

class backward_induction_class():

    def __init__(self,**kwargs):
        
        self.setup(**kwargs)

    def setup(self,**kwargs): 
        
        """ baseline parameters """

        # a. maximum number of chrildren 
        self.n = 5

        # b. number of couples 
        self.N = 50

        # b. terminal contracept age 
        self.terminal_age = 44

        # c. marriage age
        self.marriage_age = 23

        # d. number of time periods doing fertile years 
        self.T = self.terminal_age - self.marriage_age

        # e. set beta 
        self.beta = 0.95      

        # f. utility per number of children
        self.eta1 = 0.13 

        # g. utility per number of children squared 
        self.eta2 = 0.15 

        # h. disutility of contracepting  
        self.mu = -0.12  

        # i. Transition probability ([child, no child])
        #    Agent's beliefs about how the state will envolve given decision 
        self.p = np.array([0.2, 0.8])  

        #self.pk = np.random.uniform(size=self.T)

        # j. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 

        # c. Create grid
        self.create_grid()

    def create_grid(self):
        self.grid = np.arange(0, self.n) # number of chrildren grid
        # self.cost = self.eta2*self.grid + self.eta2*self.grid**2  # cost function
        # self.dc = self.grid
        self.state_transition() 

    def state_transition(self):
        
        '''Compute transition probability matrixes conditional on choice '''
        
        p = np.append(self.p,1-np.sum(self.p)) # Get transition probabilities
        P1 = np.zeros((self.n,self.n)) # Initialize transition matrix
        # Loop over rows
        for i in range(self.n):
            # Check if p vector fits entirely
            if i <= self.n-len(p):
                P1[i][i:i+len(p)]=p
            else:
                P1[i][i:] = p[:self.n-len(p)-i]
                P1[i][-1] = 1.0-P1[i][:-1].sum()

        # Conditional on d = 1, contracept
        #P2 = np.zeros((self.n,self.n))
        # Loop over rows
        #for i in range(self.n):
        #    P2[i][:len(p)]=p
        P2 = np.identity(5)
        self.P1 = P1
        self.P2 = P2
    
    def bellman(self, ev0):
        
        '''Evaluate Bellman operator, choice probability and Frechet derivative - written in integrated value form'''

        # Value of options:
        ## O: Not contracept 
        value_0 = self.eta1 + self.eta2 + self.beta * self.P1 @ ev0 # nx1 matrix
        ## 1: Contracept 
        value_1 = self.eta1 + self.eta2 + self.mu + self.beta * self.P2 @ ev0 # 1x1

        # recenter Bellman by subtracting max(VK, VR)
        maxV = np.maximum(value_0, value_1) 
        logsum = (maxV + np.log(np.exp(value_0-maxV)  +  np.exp(value_1-maxV)))  # Compute logsum to handle expectation over unobserved states
        ev1 = logsum # Bellman operator as integrated value

        # Compute choice probability of keep
        pk = 1/(1+np.exp(value_1-value_0))   

        return ev1, pk
    
    def sim_data(self, bellman, ev0, ): 
        
        """ Simulate data """

        # Set N (number of couples) and T (fertile years)
        N = self.N
        T = self.T

        # Set random seed 
        np.random.seed(2020)

        # Index 
        idx = np.tile(np.arange(1,N+1),(T,1))  
        t = np.tile(np.arange(1,T+1),(N,1)).T
            
        # Draw random numbers
        u_init = np.random.randint(1,size=(1,N)) # initial condition
        u_dx = np.random.rand(T,N)               # number of children 
        u_d = np.random.rand(T,N)                # decision/choice
        
        # Find states and choices
        csum_p = np.cumsum(self.p)               # Cumulated sum of p 
        ## this loop will iterate twice and dx1 will be incremented by either 0 or 1 on each iteration depending on the result of the comparison
        dx1 = 0
        for val in csum_p:
            dx1 += u_dx>val

        # Find states and choices
        ## state 
        x  = np.zeros((T,N), dtype=int)
        ## state next period 
        x1 = np.zeros((T,N), dtype=int)
        ## decision/choices
        d  = np.nan + np.zeros((T,N))
        ## initial condition
        x[0,:] = u_init.astype(int)
        ## Overall, this code simulates a stochastic process that models the decision to use contraception based on the current number of children (x) and a random draw (u_d), 
        ## and updates the number of children based on the decision to use contraception and a second random draw (dx1). 
        ## The pk variable is used to parameterize the probability of using contraception based on the current number of children.
        for it in range(T):
            d[it,:] = u_d[it,:] < 1-pk[x[it,:]]   # Contracept = 1 , not contracept = 0   
            if not any(d[it,:]): # if not contracept
                x1[it,:] = np.minimum((x[it,:]+1) + dx1[it,:] , self.n-1) # State transition, minimum to avoid exceeding the maximum number of children
            else: # if contracept
                x1[it,:] = np.minimum(x[it,:]*(1-d[it,:]) + dx1[it,:] , self.n-1) # State transition, minimum to avoid exceeding the maximum number of children
            
            # Ensure that the number of children cannot decrease
            x1[it,:] = np.maximum(x1[it,:], x[it,:])

            if it > 0 and np.all(x[it-1,:] == self.n-1):
                d[it,:] = 1

            if it < T-1:
                x[it+1,:] = x1[it,:]

       
        # reshape 
        idx = np.reshape(idx,T*N,order='F')
        t   = np.reshape(t,T*N,order='F')
        d   = np.reshape(d,T*N,order='F')
        x   = np.reshape(x,T*N,order='F') + 1   # add 1 to make index start at 1 as in data - 1,2,...,n
        x1  = np.reshape(x1,T*N,order='F') + 1 # add 1 to make index start at 1 as in data - 1,2,...,n
        dx1 = np.reshape(dx1,T*N,order='F')

        data = {'id': idx,'t': t, 'd': d, 'x': x, 'dx1': dx1, 'x1': x1}
        df = pd.DataFrame(data) 

        return df

    # def initialze(self):

    #     T = self.T

    #     # a. empty tables size of terminal value and T 
    #     self.Vstar_bi = np.nan + np.zeros([self.terminal_W+1,T])
    #     self.Cstar_bi = np.nan + np.zeros([self.terminal_W+1, T])

    #     # b. 
    #     self.Cstar_bi[:,T-1] = np.arange(self.terminal_W+1) 

    #     # c. utility from period T-1 
    #     self.Vstar_bi[:,T-1] = self.utility(self.Cstar_bi[:,T])
    

    # def utility(self):

    #     N = self.N
        
    #     self.u_0 = self.eta1*N + self.eta2*N**2

   