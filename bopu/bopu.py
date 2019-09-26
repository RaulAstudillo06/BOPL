# Copyright (c) 2019, Raul Astudillo

import numpy as np
import time
import os
from aux_software.GPyOpt.core.errors import InvalidConfigError
from aux_software.GPyOpt.core.task.cost import CostModel
from aux_software.GPyOpt.optimization import GeneralOptimizer


class BOPU(object):
    """
    Runner of the Bayesian-optimization-under-utility-uncertainty loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param attributes: attributes class.
    :param sampling_policy: sampling policy class.
    :param utility: utility function. See utility folder for more information.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    """

    def __init__(self, model, space, attributes, sampling_policy, utility, X_init, Y_init=None, true_underlying_utility_func=None, dynamic_utility_parameter_distribution=False):
        self.model = model
        self.space = space
        self.attributes = attributes
        self.sampling_policy = sampling_policy
        self.utility = utility
        self.X = X_init
        self.Y = Y_init
        self.true_underlying_utility_func = true_underlying_utility_func
        if dynamic_utility_parameter_distribution:
            if self.utility.parameter_distribution.elicitation_strategy is not None:
                self.dynamic_utility_parameter_distribution = True
            else:
                self.dynamic_utility_parameter_distribution = False
                print('Preference elicitation strategy has not been provided')
        else:
            self.dynamic_utility_parameter_distribution = False

        self.cost = CostModel(None)
        self.expectation_utility = utility.expectation
        self.n_attributes = self.attributes.get_output_dim()
        self.number_of_gp_hyps_samples = min(10, self.model.number_of_hyps_samples())
        self.utility_support = utility.parameter_distribution.support
        self.utility_prob_dist = utility.parameter_distribution.prob_dist
        self.full_utility_support = self.utility.parameter_distribution.use_full_support
        if self.full_utility_support:
            self.utility_support_cardinality = len(self.utility_support)

        self.historical_underlying_optima = []
        self.historical_underlying_optimal_values = []
        self.historical_integrated_optimal_values = []
        self.evaluation_optimizer = GeneralOptimizer(optimizer='lbfgs', space=self.space, parallel=False)
        script_dir = os.path.dirname(__file__)
        self.project_path = script_dir[:-5]

    def run_optimization(self, max_iter=1, filename=None, max_time=np.inf, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False, compute_true_integrated_optimal_value=False, compute_integrated_optimal_values=False):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param filename: filename of the file the optimization results are saved (default, None).
        """

        if self.attributes is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # Compute real optimum value
        if compute_true_underlying_optimal_value:
            if self.true_underlying_utility_func is not None:
                self.compute_true_underlying_optimal_value = True
                self.get_true_underlying_optimal_value(filename)
            else:
                print('Cannot compute underlying optimal values without true underlying utility function')

        if compute_integrated_optimal_values:
            if not self.dynamic_utility_parameter_distribution:
                self.compute_integrated_optimal_values = True
            else:
                print('It does not make sense to compute integrated optimal values when the utility distribution is dynamic.')

        if compute_true_integrated_optimal_value:
            if not self.dynamic_utility_parameter_distribution:
                self.get_true_integrated_optimal_value(filename)
            else:
                print('It does not make sense to compute the true integrated optimal value when the utility distribution is dynamic.')

        # Save the options to print and save the results
        self.filename = filename
        self.report_evaluated_designs_only = report_evaluated_designs_only
        self.utility_distribution_update_interval = utility_distribution_update_interval

        # Setting up stop conditions
        if (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # Initial function evaluation
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.attributes.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
        # Initialize model
        self.model.updateModel(self.X, self.Y)
        self.model.get_model_parameters_names()
        self.model.get_model_parameters()

        # Initialize iterations and running time
        self.time_zero = time.time()
        self.num_acquisitions = 0
        self.cum_time = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        # Initialize time cost of the evaluations
        while (self.max_time > self.cum_time) and (self.num_acquisitions < self.max_iter):
            if (self.num_acquisitions % self.utility_distribution_update_interval) == 0:
                self._update_utility_distribution
            self.suggested_sample = self.compute_next_evaluations()
            self.X = np.vstack((self.X, self.suggested_sample))

            # Evaluate *f* in X, augment Y and update cost function (if needed)
            print('Experiment: ' + filename[0])
            print('Sampling policy: ' + filename[1])
            print('Replication id: ' + filename[2])
            print('Acquisition number: {}'.format(self.num_acquisitions + 1))
            self.evaluate_objective()
            if filename is not None:
                self.save_evaluations(filename)

            # Update model
            self._update_model()
            self.model.get_model_parameters_names()
            self.model.get_model_parameters()
            if self.true_underlying_utility_func is not None:
                self._compute_current_underlying_max_value()
            if self.compute_integrated_optimal_values:
                self._compute_current_integrated_max_value()
            if filename is not None:
                self._save_results(filename)
            # Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

    def get_true_underlying_optimal_value(self, filename=None):
        if self.true_underlying_utility_func is not None:
            def objective_func(x):
                fx = np.squeeze(np.asarray(self.attributes.evaluate(x)[0]))
                val = np.atleast_1d(self.true_underlying_utility_func(fx))
                return -val

            optimum, optimal_value = self.evaluation_optimizer.optimize(f=objective_func, parallel=False)
            self.true_underlying_optimum = optimum
            self.true_underlying_optimal_value = -np.atleast_1d(optimal_value)
            print('True underlying real optimum: {}'.format(np.asscalar(self.true_underlying_optimal_value)))
            if filename is not None:
                results_folder_name = self.project_path + '/experiments/results/' + filename[
                    0] + '/historical_underlying_regret'
                if not os.path.exists(results_folder_name):
                    os.makedirs(results_folder_name)
                results_filename = filename[0] + '_' + filename[1] + '_underlying_true_optimal_value_' + filename[2]
                directory = results_folder_name + '/' + results_filename + '.txt'
                np.savetxt(directory, self.true_underlying_optimal_value)
        else:
            print('True underlying utility function has not been provided.')

    def get_true_integrated_optimal_value(self, filename=None):
        if not self.dynamic_utility_parameter_distribution:
            opt_val = 0.
            if self.full_utility_support:
                print('Computing true integrated optimal value:')
                for l in range(self.utility_support_cardinality):
                    print('Utility parameter: {}'.format(self.utility_support[l]))

                    def marginal_val_func(x):
                        fx = np.squeeze(np.asarray(self.attributes.evaluate(x)[0]))
                        val = np.atleast_1d(self.utility.eval_func(fx, self.utility_support[l]))
                        return -val

                    marginal_opt, marginal_opt_val = self.evaluation_optimizer.optimize(f=marginal_val_func, parallel=False)
                    marginal_opt_val = -marginal_opt_val
                    print('Marginal optimum: {}'.format(marginal_opt))
                    print('True marginal optimal value: {}'.format(np.asscalar(marginal_opt_val)))
                    opt_val += self.utility_prob_dist[l] * marginal_opt_val

                self.true_integrated_optimal_value = np.atleast_1d(opt_val)
                print('True integrated real optimum: {}'.format(np.asscalar(opt_val)))
                if filename is not None:
                    results_folder_name = self.project_path + '/experiments/results/' + filename[
                        0] + '/historical_integrated_optimal_values'
                    results_filename = filename[0] + '_' + filename[1] + '_true_integrated_optimal_value'
                    directory = results_folder_name + '/' + results_filename + '.txt'
                    np.savetxt(directory, self.true_integrated_optimal_value)

    def _compute_current_underlying_max_value(self):
        """
        """
        if self.true_underlying_utility_func is not None:
            if self.report_evaluated_designs_only:
                opt_val = -np.infty
                for i in range(self.X.shape[0]):
                    y_i = np.atleast_1d([self.Y[j][i, 0] for j in range(self.n_attributes)])
                    if opt_val < self.true_underlying_utility_func(y_i):
                        optimum = self.X[i, :]
                        opt_val = self.true_underlying_utility_func(y_i)
                self.historical_underlying_optima.append(optimum)
                self.historical_underlying_optimal_values.append(opt_val)
            if self.compute_true_underlying_optimal_value:
                print('Current underlying regret: {}'.format(np.asscalar(self.true_underlying_optimal_value - opt_val)))
            else:
                print('Current underlying optimal value: {}'.format(np.asscalar(opt_val)))
        else:
            print('True underlying utility function has not been provided.')

    def _compute_current_integrated_max_value(self):
        """
        Computes E_n[U(f(x_max))|f], where U is the utility function, f is the true underlying ojective function and x_max = argmax E_n[U(f(x))|U]. See
        function _marginal_max_value_so_far below.
        """
        val = 0.
        if self.full_utility_support:
            for l in range(self.utility_support_cardinality):
                #print('Utility parameter: {}'.format(self.utility_support[l]))
                if self.report_evaluated_designs_only:
                    marginal_opt_val = -np.infty
                    for i in range(self.X.shape[0]):
                        y_i = np.atleast_1d([self.Y[j][i, 0] for j in range(self.n_attributes)])
                        if marginal_opt_val < self.utility.eval_func(y_i, self.utility_support[l]):
                            marginal_argmax = self.X[i, :]
                            marginal_opt_val = self.utility.eval_func(y_i, self.utility_support[l])
                    #self.current_expected_marginal_best_point[l] = self._current_marginal_argmax(self.utility_support[l])
                else:
                    marginal_argmax = self._current_marginal_argmax(self.utility_support[l])
                    #self.current_expected_marginal_best_point[l] = marginal_argmax
                    f_at_marginal_max = np.reshape(self.attributes.evaluate(marginal_argmax)[0], (self.n_attributes,))
                    marginal_opt_val = self.utility.eval_func(f_at_marginal_max, self.utility_support[l])
                #print('Current marginal optimum: {}'.format(marginal_argmax))
                val += self.utility_prob_dist[l]*marginal_opt_val
            print('Current integrated optimal value: {}'.format(val))
            self.historical_integrated_optimal_values.append(val)

    def _current_marginal_max_value(self, parameter):
        marginal_argmax = self._current_marginal_argmax(parameter)
        marginal_max_val = np.reshape(self.attributes.evaluate(marginal_argmax)[0], (self.attributes.output_dim,))
        return self.utility.eval_func(marginal_max_val, parameter)

    def _current_marginal_argmax(self, parameter):
        """
        Computes argmax E_n[U(f(x))|U] (The abuse of notation can be misleading; note that the expectation is with
        respect to the posterior distribution on f after n evaluations)
        """
        if self.utility.affine:
            if self.n_attributes == 1:
                def val_func(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        valX += np.reshape(parameter*muX, (X.shape[0], 1))
                    return -valX

                def val_func_with_gradient(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    dval_dX = np.zeros(X.shape)
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        dmu_dX = self.model.posterior_mean_gradient(X)
                        valX += np.reshape(parameter*muX, (X.shape[0], 1))
                        dval_dX += np.reshape(parameter*dmu_dX, X.shape)
                    return -valX, -dval_dX
            else:              
                def val_func(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        valX += np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                    return -valX
    
                def val_func_with_gradient(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    dval_dX = np.zeros(X.shape)
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        dmu_dX = self.model.posterior_mean_gradient(X)
                        valX += np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                        dval_dX += np.tensordot(parameter, dmu_dX, axes=1)
                    return -valX, -dval_dX

        elif self.expectation_utility is not None:
            def val_func(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                for h in range(self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    for i in range(X.shape[0]):
                        func_val[i,0] += self.expectation_utility.func(parameter, mean[:,i], var[:,i])
                return -func_val

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                func_gradient = np.zeros(X.shape)
                for h in range(self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    dmean_dX = self.model.posterior_mean_gradient(X)
                    dvar_dX = self.model.posterior_variance_gradient(X)
                    aux = np.concatenate((dmean_dX,dvar_dX))
                    for i in range(X.shape[0]):
                        func_val[i,0] += self.expectation_utility.func(parameter, mean[:,i], var[:,i])
                        func_gradient[i, :] += np.matmul(self.expectation_utility.gradient(parameter,mean[:,i],var[:,i]),aux[:,i])
                return -func_val, -func_gradient

        else:
            Z_samples = np.random.normal(size=(50, self.n_attributes))

            def val_func(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                for h in range(self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    std = np.sqrt(var)
                    for i in range(X.shape[0]):
                        for Z in Z_samples:
                            func_val[i,0] += self.utility.eval_func(mean[:,i] + np.multiply(std[:,i],Z), parameter)
                return -func_val

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                func_gradient = np.zeros(X.shape)
                for h in range(self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    std = np.sqrt(var)
                    dmean_dX = self.model.posterior_mean_gradient(X)
                    dstd_dX = self.model.posterior_variance_gradient(X)
                    for i in range(X.shape[0]):
                        for j in range(self.n_attributes):
                            dstd_dX[j, i, :] /= (2*std[j,i])
                        for Z in Z_samples:
                            aux1 = mean[:, i] + np.multiply(Z, std[:,i])
                            func_val[i, 0] += self.utility.eval_func(aux1, parameter)
                            aux2 = dmean_dX[:, i, :] + np.multiply(dstd_dX[:, i, :].T, Z).T
                            func_gradient[i, :] += np.matmul(self.utility.eval_gradient(aux1, parameter), aux2)
                return -func_val, -func_gradient

        argmax = self.evaluation_optimizer.optimize(f=val_func, f_df=val_func_with_gradient, parallel=False)[0]
        return argmax

    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print('Suggested point to evaluate: {}'.format(self.suggested_sample))
        self.Y_new, cost_new = self.attributes.evaluate_w_noise(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        for j in range(self.n_attributes):
            self.Y[j] = np.vstack((self.Y[j], self.Y_new[j]))

    def compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        return self.sampling_policy.suggest_sample()

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        # Input that goes into the model (is unziped in case there are categorical variables)
        X_inmodel = self.space.unzip_inputs(self.X)
        Y_inmodel = list(self.Y)
        self.model.updateModel(X_inmodel, Y_inmodel)

    def _update_utility_distribution(self):
        """
        """
        self.utility.update_parameter_distribution(self.true_underlying_utility_func, self.Y)

    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))

    def _perturb(self, x):
        perturbed_x = np.copy(x)
        while np.all(perturbed_x == x):
            perturbed_x = x + np.random.normal(size=x.shape, scale=1e-2)
            perturbed_x = self.space.round_optimum(perturbed_x)

        return perturbed_x

    def save_evaluations(self, filename):
        """
        """
        experiment_folder_name = self.project_path + '/experiments/results/' + filename[0]
        experiment_name = filename[0] + '_' + filename[1] + '_' + filename[2]
        directory = experiment_folder_name + '/X'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(directory + '/' + experiment_name + '_X.txt', self.X)
        directory = experiment_folder_name + '/Y'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(directory + '/' + experiment_name + '_Y.txt', np.squeeze(np.asarray(self.Y)))

    def _save_results(self, filename):
        """
        """
        if self.compute_true_underlying_optimal_value:
            results_folder_name = self.project_path + '/experiments/results/' + filename[
                0] + '/historical_underlying_regret'
            if not os.path.exists(results_folder_name) :
                os.makedirs(results_folder_name)
            results_filename = filename[0] + '_' + filename[1] + '_underlying_regret_' + filename[2]
            results = self.true_underlying_optimal_value - np.atleast_1d(self.historical_underlying_optimal_values)
        else:
            results_folder_name = self.project_path + '/experiments/results/' + filename[
                0] + '/historical_underlying_optimal_values'
            if not os.path.exists(results_folder_name):
                os.makedirs(results_folder_name)
            results_filename = filename[0] + '_' + filename[1] + '_underlying_optimal_values_' + filename[2]
            results = np.atleast_1d(self.historical_underlying_optimal_values)
        directory = results_folder_name + '/' + results_filename + '.txt'
        np.savetxt(directory, results)

        if self.compute_integrated_optimal_values:
            results_folder_name = self.project_path + '/experiments/results/' + filename[
                0] + '/historical_integrated_optimal_values'
            if not os.path.exists(results_folder_name):
                os.makedirs(results_folder_name)
            results_filename = filename[0] + '_' + filename[1] + '_integrated_optimal_values_' + filename[2]
            directory = results_folder_name + '/' + results_filename + '.txt'
            results = np.atleast_1d(self.historical_integrated_optimal_values)
            np.savetxt(directory, results)
        #aux_filename = experiment_folder_name + '/historical marginal optima/' + experiment_name + '_historical_marginal_optima.txt'
        #np.savetxt(aux_filename, self.historical_marginal_optima)

