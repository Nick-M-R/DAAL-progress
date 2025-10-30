"""
MOOFoil.basis

A collection of airfoil geometry constraint class.

A constraint class behaves as:
1. Initialized in problem statement 
2. Takes in input of x, z coordinates
3. Returns evaluated constraint violation
    
""" 

import numpy as np

class Constraint():
    def __init__(self, cases, key, lower_bound = -1e16, upper_bound = 1e16):
        self.key = key
        self.cases = cases
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def eval(self, sol):
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        
        try:
            g = []
            for case in self.cases:
                f = sol[case][self.key]  
                g.append(
                    # (f <= lower_bound)*(lower_bound-f)
                    # + (f >= upper_bound)*(f-upper_bound) 
                    max(lower_bound-f,f-upper_bound)
                )
            return g
        except:
            return np.ones(len(self.cases))
