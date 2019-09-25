# C# Copyright (c) 2019, Raul Astudillo Marban

import numpy as np

class ExpectationUtility(object):
    """

    """

    def __init__(self, func, gradient):
        self.func = func
        self.gradient = gradient
        
    def evaluate_func_and_gradient(self, mean, var, theta):
        """
        Samples random parameter from parameter distribution and evaluates the utility function and its gradient at y given this parameter.
        """
        EU = self.eval_func(mean, var, theta)
        dEU = self._eval_gradient(mean, var, theta)
        return U, dU
    
    
    def eval_func(self, mean, var, theta):
        """
        Evaluates the utility function at y given a fixed parameter.
        """
        return self.func(mean, var, theta)
    
    
    def eval_gradient(self, mean, var, theta):
        """
        Evaluates the gradient f the utility function at y given a fixed parameter.
        """
        return self.gradient(mean, var, theta)
