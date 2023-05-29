import numpy as np
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
        self.p1 = np.array([0.3, 0.7]) 
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


    def read_data(): 
        # data = np.loadtxt(open("carro-mira.csv"), delimiter=",")

        data = pd.read_csv('carro-mira_new.txt', sep=' ', header=None)

        edu_h = data[:,1]           # education of husband
        edu_w = data[:,2]           # education of wife
        age_w_m = data[:,3]         # age of wife at the moment of marriage
        idx = data[:,4]             # couple id
        t = data[:,5]               # year
        age_w = data[:,6]           # age of wife
        age_h = data[:,7]           # age of husband
        c_birth_i = data[:,8]       # current birth indicator
        n_birth_i = data[:,9]       # next period birth indicator
        x = data[:,10]              # number of children
        cc = data[:,11]             # current contraception choice (1, 2, 3)
        p_birth_i = data[:,12]      # previous birth indicator
        r = data[:,13]              # religious couple indicator 
        no_c = data [:,14]          # no contraception (dummy)
        c = data[:,15]              # contraception (dummy)
        s_c = data[:,16]            # sterilization (dummy)
        p_c = data[:,25]            # previous contraception choice


        # change type to integrer
        x = x.astype(int)
        dx1 = dx1.astype(int)

        # Collect in a dataframe
        data = {
                'edu_h': edu_h,
                'edu_w': edu_w,
                'age_w_m': age_w_m,
                'idx': idx,
                'year': t,
                'age_w': age_w,
                'age_h': age_h,
                'c_birth_i': c_birth_i,
                'n_birth_i': n_birth_i,
                'x': x,
                'cc': cc,
                'p_birth_i': p_birth_i,
                'r': r,
                'no_c': no_c,
                'c': c,
                's_c': s_c,
                'p_c': p_c
                }
        
        # as dataframe 
        df = pd.DataFrame(data) 

        # Remove observations with missing lagged mileage
        # df = df.drop(df[df['contraception choice'] == 3].index, axis=0)
        # df = df.drop(df[df['x'] > 4].index, axis=0)

        # save data
        # dta = df.drop(['contraception choice'],axis=1)
        
        return df