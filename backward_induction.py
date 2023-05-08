from types import SimpleNamespace
import numpy as np

def backward_induction():

    def __init__(self,**kwargs):
        
        self.setup(**kwargs)

    def setup(self,**kwargs): 
        """ baseline parameters """

        # a. maximum number of chrildren 
        self.child_max = 5

        # b. terminal contracept age 
        self.terminal_age = 44

        # c. marriage age
        self.marriage_age = 23

        # d. number of time periods doing fertile years 
        self.simT = self.terminal_age - self.marriage_age

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

        # j. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 

    def initialze(self):

        T = self.simT

        # a. empty tables size of terminal value and T 
        self.Vstar_bi = np.nan + np.zeros([self.terminal_W+1,T])
        self.Cstar_bi = np.nan + np.zeros([self.terminal_W+1, T])

        # b. 
        self.Cstar_bi[:,T-1] = np.arange(self.terminal_W+1) 

        # c. utility from period T-1 
        self.Vstar_bi[:,T-1] = self.utility(self.Cstar_bi[:,T])
    
    def utility(self, N):
        
        self.u_0 = self.eta1*N + self.eta2*N**2
        






        
        

    












