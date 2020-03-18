# Copyright (c) 2019, Raul Astudillo Marban

import numpy as np
from scipy.stats import norm
from aux_software.GPyOpt.acquisitions.base import AcquisitionBase


class uEI_constrained(AcquisitionBase):
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
        super(uEI_constrained, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.use_full_support = self.utility.parameter_distribution.use_full_support
        self.number_of_gp_hyps_samples = 10
        if self.use_full_support:
            self.utility_parameter_samples = self.utility.parameter_distribution.support
            self.utility_prob_dist = self.utility.parameter_distribution.prob_dist
        else:
            self.utility_parameter_samples = self.utility.sample_parameter(10)

    def _compute_acq(self, X):
        """
        Computes the Expected Improvement per unit of cost
        """
        X = np.atleast_2d(X)
        marginal_acqX = self._marginal_acq(X, self.utility_parameter_samples)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(self.utility_parameter_samples)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        return acqX

    def _marginal_acq(self, X, utility_parameter_samples):
        L = len(utility_parameter_samples)
        marginal_acqX = np.zeros((X.shape[0], L))
        marginal_max_feasible_val_so_far = self._marginal_max_feasible_val_so_far(utility_parameter_samples)
        n_h = self.number_of_gp_hyps_samples  # Number of GP hyperparameters samples.
        for h in range(n_h):
            self.model.set_hyperparameters(h)
            meanX, varX = self.model.predict(X)
            sigmaX = np.sqrt(varX)
            for l in range(L):
                for i in range(X.shape[0]):
                    mu = meanX[0, i]
                    sigma = sigmaX[0, i]
                    phi = norm.pdf((mu - marginal_max_feasible_val_so_far[l]) / sigma)
                    Phi = norm.cdf((mu - marginal_max_feasible_val_so_far[l]) / sigma)
                    val = (mu - marginal_max_feasible_val_so_far[l]) * Phi + sigma * phi
                    for j in range(1, meanX.shape[0]):
                        mu = meanX[j, i]
                        sigma = sigmaX[j, i]
                        Phi = norm.cdf((mu - utility_parameter_samples[l][j-1]) / sigma)
                        val *= Phi
                    marginal_acqX[i, l] += val

        marginal_acqX = marginal_acqX / n_h
        return marginal_acqX
    
    def _compute_acq_withGradients(self, X):
        """
        """
        X = np.atleast_2d(X)
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples
        utility_parameter_samples2 = self.utility_parameter_samples
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, utility_parameter_samples2)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
            dacq_dX = np.tensordot(marginal_dacq_dX, self.utility_prob_dist, 1)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(utility_parameter_samples2)
            dacq_dX = np.sum(marginal_dacq_dX, axis=2) / len(utility_parameter_samples2)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        return acqX, dacq_dX
    
    def _marginal_acq_with_gradient(self, X, utility_parameter_samples):
        L = len(utility_parameter_samples)
        marginal_acqX = np.zeros((X.shape[0],L))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], L))
        marginal_max_feasible_val_so_far = self._marginal_max_feasible_val_so_far(utility_parameter_samples)
        n_h = self.number_of_gp_hyps_samples # Number of GP hyperparameters samples.
        for h in range(n_h):
            self.model.set_hyperparameters(h)
            meanX, varX = self.model.predict(X)
            sigmaX = np.sqrt(varX)
            dmean_dX = self.model.posterior_mean_gradient(X)
            dvar_dX = self.model.posterior_variance_gradient(X)
            for l in range(L):
                for i in range(X.shape[0]):
                    mu = meanX[0, i]
                    sigma = sigmaX[0, i]
                    phi = norm.pdf((mu - marginal_max_feasible_val_so_far[l]) / sigma)
                    Phi = norm.cdf((mu - marginal_max_feasible_val_so_far[l]) / sigma)
                    val = (mu - marginal_max_feasible_val_so_far[l]) * Phi + sigma * phi
                    val_aux = np.copy(val)
                    dsigma_dX = (0.5 / sigma) * dvar_dX[0, i, :]
                    gradient = dmean_dX[0, i, :] * Phi + phi * dsigma_dX 
                    for j in range(1, meanX.shape[0]):
                        mu = meanX[j, i]
                        sigma = sigmaX[j, i]
                        Phi = norm.cdf((mu - utility_parameter_samples[l][j-1]) / sigma)
                        val *= Phi
                        gradient *= Phi
                    gradient += val_aux * aux_function_for_gradient(meanX[1:, i], varX[1:, i], dmean_dX[1:, i, :], dvar_dX[1:, i, :], utility_parameter_samples[l])
                    marginal_acqX[i, l] += val
                    marginal_dacq_dX[i, :, l] += gradient
                    
        marginal_acqX = marginal_acqX/n_h          
        marginal_dacq_dX = marginal_dacq_dX/n_h
        return marginal_acqX, marginal_dacq_dX    

    def update_samples(self):
        print('Update utility parameter samples')
        if self.use_full_support:
            self.utility_parameter_samples = self.utility.parameter_distribution.support
            self.utility_prob_dist = self.utility.parameter_distribution.prob_dist
        else:
            self.utility_parameter_samples = self.utility.sample_parameter(10)

    def _marginal_max_feasible_val_so_far(self, utility_parameter_samples):
        """

        :return:
        """
        L = len(utility_parameter_samples)
        marginal_max_feasible_val_so_far = np.empty((L, ))
        fX_evaluated = self.model.posterior_mean_at_evaluated_points()
        for l in range(L):
            marginal_max_feasible_val_so_far[l] = np.max(self.utility.eval_func(fX_evaluated, utility_parameter_samples[l]))
        return marginal_max_feasible_val_so_far
    
    
def aux_function_for_gradient(mean, var, mean_gradient, var_gradient, utility_parameter_sample):
    j = len(mean)
    mu = mean[j - 1]
    sigma = np.sqrt(var[j - 1])
    output = norm.pdf((mu - utility_parameter_sample[j - 1]) / sigma) * ((mean_gradient[j - 1, :] / sigma ) - (0.5 * (mu - utility_parameter_sample[j - 1]) / np.power(var[j - 1], 1.5)) * var_gradient[j - 1, :])
    if j == 1:
        return output
    else:
        Phi_aux = norm.cdf((mu - utility_parameter_sample[j - 1]) / sigma)
        for k in range(j - 1):
            mu = mean[k]
            sigma = np.sqrt(var[k])
            Phi = norm.cdf((mu - utility_parameter_sample[k]) / sigma)
            output *= Phi
        output += Phi_aux * aux_function_for_gradient(mean[:j - 1], var[:j - 1], mean_gradient[:j - 1, :], var_gradient[:j - 1, :], utility_parameter_sample[:j - 1])
        return output
