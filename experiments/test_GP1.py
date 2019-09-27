if __name__ == '__main__':
    import sys
    sys.path.append('/home/raul/Projects/BOPU/bopu')
    import numpy as np
    import scipy
    import aux_software.GPyOpt as GPyOpt
    import aux_software.GPy as GPy
    from core import Attributes
    from core import MultiOutputGP
    from sampling_policies import Random
    from sampling_policies import AcquisitionFunction
    from sampling_policies.acquisition_functions import uEI_affine
    from utility import UtilityDistribution
    from utility import Utility
    from utility import ExpectationUtility
    from utility.elicitation_strategies import random_preference_elicitation
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer
    import scipy.optimize as optimize

    # Function to optimize
    d = 3
    I = np.linspace(0., 1., 10)
    x, y, z = np.meshgrid(I, I, I)
    grid = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    kernel = GPy.kern.Matern52(input_dim=d, variance=2., ARD=True, lengthscale=np.atleast_1d([0.3]*d))
    cov = kernel.K(grid)
    mean = np.zeros((1000,))
    r1 = np.random.RandomState(2312)
    Y1 = r1.multivariate_normal(mean, cov)
    r2 = np.random.RandomState(22)
    Y2 = r2.multivariate_normal(mean, cov)
    Y1 = np.reshape(Y1, (1000, 1))
    Y2 = np.reshape(Y2, (1000, 1))
    #print(Y1[:5, 0])
    #print(Y2[:5, 0])
    model1 = GPy.models.GPRegression(grid, Y1, kernel, noise_var=1e-10)
    model2 = GPy.models.GPRegression(grid, Y2, kernel, noise_var=1e-10)

    def f1(X):
        X_copy = np.atleast_2d(X)
        return model1.posterior_mean(X_copy)

    def f2(X):
        X_copy = np.atleast_2d(X)
        return model2.posterior_mean(X_copy)

    # noise_var = [0.25,0.25]
    attributes = Attributes([f1, f2])
    # f = MultiObjective([f1,f2], noise_var=noise_var)

    # Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Model (Multi-output GP)
    n_attributes = 2
    model = MultiOutputGP(output_dim=n_attributes, exact_feval=[True, True], fixed_hyps=False)
    # model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

    # Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1))

    # Utility function
    def utility_func(y, parameter):
        return np.dot(parameter, y)

    def utility_gradient(y, parameter):
        return parameter
    
    L = 10
    utility_parameter_support = np.empty((L, 2))
    aux = (np.pi * np.arange(L)) / (2 * (L - 1))
    utility_parameter_support[:, 0] = np.cos(aux)
    utility_parameter_support[:, 1] = np.sin(aux)
    print(np.sum(utility_parameter_support ** 2, axis=1))
    utility_parameter_prob_distribution = np.ones((L,)) / L
    utility_param_distribution = UtilityDistribution(support=utility_parameter_support, prob_dist=utility_parameter_prob_distribution, utility_func=utility_func, elicitation_strategy=random_preference_elicitation)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_param_distribution, affine=True)

    # --- Sampling policy
    sampling_policy_name = 'uEI'
    if sampling_policy_name is 'uEI':
        # Acquisition optimizer
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='lbfgs', inner_optimizer='lbfgs')

        acquisition = uEI_affine(model, space, optimizer=acquisition_optimizer, utility=utility)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        sampling_policy = AcquisitionFunction(model, space, acquisition, evaluator)
    elif sampling_policy_name is 'TS':
        sampling_policy = TS(model, optimization_space, optimizer='CMA', scenario_distribution=scenario_distribution,
                             utility=utility, expectation_utility=expectation_utility)
        acquisition = None
    elif sampling_policy_name is 'Random':
        sampling_policy = Random(model, space)

    # Compute real optimum value
    compute_real_optimum = False
    if compute_real_optimum:
        bounds = [(0, 1)] * d
        starting_points = np.random.rand(100, d)
        opt_val = 0.
        for theta in utility_parameter_support:
            def marginal_value_func(x):
                x_copy = np.atleast_2d(x)
                fx = np.empty((2,))
                fx[0] = f1(x_copy)[0, 0]
                fx[1] = f2(x_copy)[0, 0]
                val = utility_func(fx, theta)
                return -val

            best_val_found = np.inf
            for d0 in starting_points:
                res = scipy.optimize.fmin_l_bfgs_b(marginal_value_func, d0, approx_grad=True, bounds=bounds)
                if best_val_found > res[1]:
                    best_val_found = res[1]
                    marginal_opt = res[0]

            print('Utility parameter: {}'.format(theta))
            print('Marginal optimum: {}'.format(marginal_opt))
            print('Marginal optimal value: {}'.format(-np.asscalar(best_val_found)))
            opt_val -= best_val_found
        opt_val /= len(utility_parameter_support)
        print('Integrated real optimum: {}'.format(np.asscalar(opt_val)))

    # BO model
    max_iter = 100
    experiment_name = 'test_GP1'
    if len(sys.argv) > 1:
        experiment_number = str(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, experiment_number]

        # True underlying utility
        true_underlying_utility_parameter = utility.sample_parameter()
        print(true_underlying_utility_parameter)

        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)

        bopu = BOPU(model, space, objective, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=True)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)

    else:
        for i in range(1):
            experiment_number = str(i)
            filename = [experiment_name, sampling_policy_name, experiment_number]

            # True underlying utility
            true_underlying_utility_parameter = utility.sample_parameter()
            print(true_underlying_utility_parameter)

            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)

            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=True)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)

