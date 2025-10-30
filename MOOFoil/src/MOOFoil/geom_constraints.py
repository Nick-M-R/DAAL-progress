"""
MOOFoil.basis

A collection of airfoil geometry constraint class.

A constraint class behaves as:
1. Initialized in problem statement 
2. Takes in input of x, z coordinates
3. Returns evaluated constraint violation
    
""" 

import numpy as np

class MaxThickness:
    def __init__(self, lower_bound = 0, upper_bound = 1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def eval(self, af):
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        t_max = af.max_thickness()[0]
        return (
            (t_max <= lower_bound)*(lower_bound-t_max)
            + (t_max >= upper_bound)*(t_max-upper_bound) 
        )
        
class MinThickness:
    def __init__(self, lower_bound = 0, upper_bound = 1, bounds = (0.1, 0.9)):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bounds = bounds
        
    def eval(self, af):
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        t_min = af.min_thickness(bounds = self.bounds)[0]
        # print(t_min)
        return (
            (t_min <= lower_bound)*(lower_bound-t_min)
            + (t_min >= upper_bound)*(t_min-upper_bound) 
        )
        
