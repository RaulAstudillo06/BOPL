if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(script_dir[:-11] + 'bopu')
    import numpy as np
    import scipy
    import aux_software.GPyOpt as GPyOpt
    import aux_software.GPy as GPy
    from core import Attributes
    from models import MultiOutputGP
    from models import BasicModel
    from sampling_policies import Random
    from sampling_policies import AcquisitionFunction
    from sampling_policies.acquisition_functions import uEI
    from utility import UtilityDistribution
    from utility import Utility
    from utility import ExpectationUtility
    from utility.elicitation_strategies import random_preference_elicitation
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer

    # Function to optimize
    d = 4
    m = 6  # Number of attributes
    aux_model = []
    I = np.linspace(0., 1., 6)
    aux_grid = np.meshgrid(I, I, I, I)
    grid = np.array([a.flatten() for a in aux_grid]).T
    kernel = GPy.kern.Matern52(input_dim=d, variance=2., ARD=True, lengthscale=np.atleast_1d([0.3]*d))
    cov = kernel.K(grid)
    mean = np.zeros((6 ** d,))
    for j in range(m):
        r = np.random.RandomState(j+7)
        Y = r.multivariate_normal(mean, cov)
        Y = np.reshape(Y, (6 ** d, 1))
        aux_model.append(GPy.models.GPRegression(grid, Y, kernel, noise_var=1e-10)) 
    
    def f(X):
        X = np.atleast_2d(X)
        fX = np.empty((m, X.shape[0]))
        for j in range(m):
            fX[j, :] = aux_model[j].posterior_mean(X)[:, 0]
        return fX

    attributes = Attributes(f, as_list=False, output_dim=m)

    # Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Utility function
    def utility_func(y, parameter):
        aux = (y.transpose() - parameter).transpose()
        return -np.sum(np.square(aux), axis=0)
    
    def utility_gradient(y, parameter):
        y_aux = np.squeeze(y)
        return -2*(y_aux - parameter)
    
        #  Parameter distribution
    X = np.zeros((2*m, d))
    bounds = [(0, 1)] * d
    starting_points = np.random.rand(100, d)
    for j in range(m):
        def marginal_func(x):
            x_copy = np.atleast_2d(x)
            val = aux_model[j].posterior_mean(x_copy)[:, 0]
            return -val
    
        best_val_found = np.inf
        for x0 in starting_points:
            res = scipy.optimize.fmin_l_bfgs_b(marginal_func, x0, approx_grad=True, bounds=bounds)
            if best_val_found > res[1]:
                best_val_found = res[1]
                marginal_opt = res[0]
        X[j, :] = marginal_opt

        def marginal_func(x):
            x_copy = np.atleast_2d(x)
            val = aux_model[j].posterior_mean(x_copy)[:, 0]
            return val

        best_val_found = np.inf
        for x0 in starting_points:
            res = scipy.optimize.fmin_l_bfgs_b(marginal_func, x0, approx_grad=True, bounds=bounds)
            if best_val_found > res[1]:
                best_val_found = res[1]
                marginal_opt = res[0]
        X[j + m, :] = marginal_opt
    utility_parameter_support = f(X).T
    utility_parameter_prob_distribution = np.ones((2*m,)) / (2*m)
    utility_param_distribution = UtilityDistribution(support=utility_parameter_support, prob_dist=utility_parameter_prob_distribution, utility_func=utility_func, elicitation_strategy=random_preference_elicitation)
    
        # Expectation of utility
    def expectation_utility_func(mu, var, parameter):
        aux = (mu.transpose() - parameter).transpose()
        val = -np.sum(np.square(aux), axis=0) - np.sum(var, axis=0)
        return val
    
    def expectation_utility_gradient(mu, var, parameter):
        mu_aux = np.squeeze(mu)
        var_aux = np.squeeze(var)
        gradient = -np.concatenate((2*(mu_aux - parameter), np.ones((len(var_aux), ))))
        return gradient
    
    expectation_utility = ExpectationUtility(expectation_utility_func, expectation_utility_gradient)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_param_distribution, expectation=expectation_utility)

    # --- Sampling policy
    sampling_policy_name = 'Random'
    if sampling_policy_name is 'uEI':
        # Model (Multi-output GP)
        model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)

        # Sampling policy
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, expectation_utility=expectation_utility, optimizer='lbfgs', inner_optimizer='lbfgs')

        acquisition = uEI(model, space, optimizer=acquisition_optimizer, utility=utility)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        sampling_policy = AcquisitionFunction(model, space, acquisition, evaluator)
    elif sampling_policy_name is 'TS':
        model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)  # Model (Multi-output GP)

        sampling_policy = TS(model, optimization_space, optimizer='CMA', scenario_distribution=scenario_distribution,
                             utility=utility, expectation_utility=expectation_utility)
    elif sampling_policy_name is 'Random':
        model = BasicModel(output_dim=m)
        sampling_policy = Random(model=None, space=space)

    # BO model
    max_iter = 100
    experiment_name = 'test_GP2'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])

        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)
        print(initial_design)

        # True underlying utility
        true_underlying_utility_parameter = utility.sample_parameter_from_prior(1, experiment_number)
        print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))

        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)

        #
        experiment_number = str(experiment_number)
        filename = [experiment_name, sampling_policy_name, experiment_number]

        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=True)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
    else:
        for i in range(1, 50):
            experiment_number = i

            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

            # True underlying utility
            true_underlying_utility_parameter = utility.sample_parameter_from_prior(1, experiment_number)
            print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))

            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)

            #
            experiment_number = str(experiment_number)
            filename = [experiment_name, sampling_policy_name, experiment_number]

            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=True)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
