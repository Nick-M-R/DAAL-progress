"""
MOOFoil.basis

A collection of airfoil geometry constraint class.

A constraint class behaves as:
1. Initialized in problem statement 
2. Takes in input of x, z coordinates
3. Returns evaluated constraint violation
    
""" 

import numpy as np

class Objective():
    def __init__(self, cases, key, normalization = 1):
        if key[0]=='-':
            self.mult = -1
            key = key[1:]
        else:
            self.mult = 1
        self.mult *= 1/normalization
        self.key = key
        self.cases = cases
        
    def eval(self, sol):
        try:
            return [self.mult*sol[case][self.key] for case in self.cases]
        except:
            return np.ones(len(self.cases))
