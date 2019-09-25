# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np
from scipy.stats import norm
from scipy.special import erfc
from aux_software.GPyOpt.acquisitions.base import AcquisitionBase
from aux_software.GPyOpt.core.task.cost import constant_cost_withGradients


class uEI_affine(AcquisitionBase):
    """
    Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, utility=None):
        self.optimizer = optimizer
        self.utility = utility
        super(uEI_affine, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients
        self.use_full_support = self.utility.parameter_distribution.use_full_support
        self.number_of_gp_hyps_samples = 10


    def _compute_acq(self, X):
        """
        Computes the Expected Improvement per unit of cost
        """
        if self.use_full_support:
            self.utility_parameter_samples = self.utility.parameter_distribution.support
            self.utility_parameter_distribution = np.atleast_1d(self.utility.parameter_distribution.prob_dist)
        else:
            self.utility_parameter_samples = self.utility.parameter_distribution.sample(20)
        X = np.atleast_2d(X)
        marginal_acqX = self._marginal_acq(X, self.utility_parameter_samples)
        if self.use_full_support:
             acqX = np.matmul(marginal_acqX, self.utility_parameter_distribution)
        else:
            acqX = np.sum(marginal_acqX, axis=1)/len(self.utility_parameter_samples)
        acqX = np.reshape(acqX, (X.shape[0],1))
        return acqX


    def _compute_acq_withGradients(self, X):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        if self.use_full_support:
            self.utility_parameter_samples = self.utility.parameter_distribution.support
            self.utility_param_dist = np.atleast_1d(self.utility.parameter_distribution.prob_dist)
        else:
            self.utility_parameter_samples = self.utility.parameter_distribution.sample()
        X = np.atleast_2d(X)
         
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, self.utility_parameter_samples)           
        if self.use_full_support:
             acqX = np.matmul(marginal_acqX, self.utility_param_dist)
             dacq_dX = np.tensordot(marginal_dacq_dX, self.utility_param_dist, 1)
        else:
            acqX = np.sum(marginal_acqX, axis=1)/len(self.utility_parameter_samples)
            dacq_dX = np.sum(marginal_dacq_dX, axis=2)/len(self.utility_parameter_samples)
            
        acqX = np.reshape(acqX,(X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        return acqX, dacq_dX
    
    
    def _marginal_acq(self, X, utility_parameter_samples):
        L = len(utility_parameter_samples)
        marginal_acqX = np.zeros((X.shape[0],L))
        n_h = self.number_of_gp_hyps_samples # Number of GP hyperparameters samples.
        for h in range(n_h):
            self.model.set_hyperparameters(h)
            meanX, varX = self.model.predict(X)
            marginal_best_so_far = self._marginal_best_so_far(utility_parameter_samples)
            for l in range(L):
                current_best = marginal_best_so_far[l]
                for i in range(X.shape[0]):
                    mu = np.dot(utility_parameter_samples[l], meanX[:,i])
                    sigma = np.sqrt(np.dot(np.square(utility_parameter_samples[l]), varX[:,i]))
                    #a = (mu-current_best)*norm.cdf((mu-current_best)/sigma) + sigma*norm.pdf((mu-current_best)/sigma)
                    phi, Phi, u = self._get_quantiles(current_best, mu, sigma)
                    marginal_acqX[i,l] += sigma*(u*Phi + phi)               
        marginal_acqX = marginal_acqX/n_h
        return marginal_acqX
    
    
    def _marginal_acq_with_gradient(self, X, utility_parameter_samples):
        L = len(utility_parameter_samples)
        marginal_acqX = np.zeros((X.shape[0],L))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], L))
        n_h = self.number_of_gp_hyps_samples # Number of GP hyperparameters samples.
        for h in range(n_h):
            self.model.set_hyperparameters(h)
            meanX, varX = self.model.predict(X)
            dmean_dX = self.model.posterior_mean_gradient(X)
            dvar_dX = self.model.posterior_variance_gradient(X)
            marginal_best_so_far = self._marginal_best_so_far(utility_parameter_samples)
            for l in range(L):
                best = marginal_best_so_far[l]
                for i in range(X.shape[0]):
                    mu = np.dot(utility_parameter_samples[l], meanX[:,i])
                    sigma = np.sqrt(np.dot(np.square(utility_parameter_samples[l]), varX[:,i]))
                    phi = norm.pdf((mu-best)/sigma)
                    Phi = norm.cdf((mu-best)/sigma)
                    marginal_acqX[i,l] += (mu-best)*Phi + sigma*phi
                    dmu_dX = np.matmul(utility_parameter_samples[l], dmean_dX[:,i,:])
                    dsigma_dX = 0.5*np.matmul(np.square(utility_parameter_samples[l]), dvar_dX[:,i,:])/sigma
                    marginal_dacq_dX[i, :, l] += dmu_dX*Phi + phi*dsigma_dX 
                    
        marginal_acqX = marginal_acqX/n_h          
        marginal_dacq_dX = marginal_dacq_dX/n_h
        return marginal_acqX, marginal_dacq_dX
    
    
    def _marginal_best_so_far(self, utility_parameter_samples):
        L = len(utility_parameter_samples)
        marginal_best = np.empty(L)
        muX_eval = self.model.posterior_mean_at_evaluated_points()
        for l in range(L):
            marginal_best[l] = max(np.matmul(utility_parameter_samples[l], muX_eval))
            
        return marginal_best
                    
            
    def _get_utility_parameters_samples(self, n_samples=None):
        if n_samples == None:
            samples = self.utility.parameter_distribution.support
        else:
            samples = self.utility.parameter_distribution.sample(n_samples)      
        return samples       
        
    
    def _get_quantiles(self, fmax, m, s):
        '''
        Quantiles of the Gaussian distribution useful to determine the acquisition function values
        :param acquisition_par: parameter of the acquisition function
        :param fmin: current minimum.
        :param m: vector of means.
        :param s: vector of standard deviations.
        '''
        if isinstance(s, np.ndarray):
            s[s < 1e-10] = 1e-10
        elif s < 1e-10:
            s = 1e-10
        u = (m-fmax)/s
        phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
        Phi = 0.5 * erfc(-u / np.sqrt(2))
        return (phi, Phi, u)
