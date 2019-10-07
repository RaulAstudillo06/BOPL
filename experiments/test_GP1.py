if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(script_dir[:-11] + 'bopu')
    import numpy as np
    import aux_software.GPyOpt as GPyOpt
    import aux_software.GPy as GPy
    from core import Attributes
    from core import MultiOutputGP
    from sampling_policies import Random
    from sampling_policies import AcquisitionFunction
    from sampling_policies.acquisition_functions import uEI_affine
    from sampling_policies import uTS
    from sampling_policies import ParEGO
    from utility import UtilityDistribution
    from utility import Utility
    from utility import ExpectationUtility
    from utility.elicitation_strategies import random_preference_elicitation
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer

    # Function to optimize
    d = 3
    m = 2
    I = np.linspace(0., 1., 10)
    x, y, z = np.meshgrid(I, I, I)
    grid = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    kernel = GPy.kern.Matern52(input_dim=d, variance=2., ARD=True, lengthscale=np.atleast_1d([0.3]*d))
    cov = kernel.K(grid)
    mean = np.zeros((1000,))
    r1 = np.random.RandomState(1)
    Y1 = r1.multivariate_normal(mean, cov)
    r2 = np.random.RandomState(2)
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
    model = MultiOutputGP(output_dim=m, exact_feval=[True, True], fixed_hyps=False)
    # model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

    # Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1))

    # Utility function
    def utility_func(y, parameter):
        return np.dot(parameter, y)

    def utility_gradient(y, parameter):
        return parameter
    
    #L = 10
    #utility_parameter_support = np.empty((L, 2))
    #aux = (np.pi * np.arange(L)) / (2 * (L - 1))
    #utility_parameter_support[:, 0] = np.cos(aux)
    #utility_parameter_support[:, 1] = np.sin(aux)
    #print(np.sum(utility_parameter_support ** 2, axis=1))
    #utility_parameter_prob_distribution = np.ones((L,)) / L

    def prior_sample_generator(n_samples):
        samples = np.empty((n_samples, 2))
        alpha = 0.5 * np.pi * np.random.rand(n_samples)
        samples[:, 0] = np.cos(alpha)
        samples[:, 1] = np.sin(alpha)
        return samples


    utility_param_distribution = UtilityDistribution(prior_sample_generator=prior_sample_generator,
                                                     utility_func=utility_func,
                                                     elicitation_strategy=random_preference_elicitation)
    #utility_param_distribution = UtilityDistribution(support=utility_parameter_support, prob_dist=utility_parameter_prob_distribution, utility_func=utility_func, elicitation_strategy=random_preference_elicitation)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_param_distribution, affine=True)

    # --- Sampling policy
    sampling_policy_name = 'ParEGO'
    if sampling_policy_name is 'uEI':
        # Acquisition optimizer
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='lbfgs', inner_optimizer='lbfgs')

        acquisition = uEI_affine(model, space, optimizer=acquisition_optimizer, utility=utility)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        sampling_policy = AcquisitionFunction(model, space, acquisition, evaluator)
    elif sampling_policy_name is 'uTS':
        sampling_policy = uTS(model, space, optimizer='CMA', utility=utility)
        acquisition = None
    elif sampling_policy_name is 'Random':
        sampling_policy = Random(model, space)
    elif sampling_policy_name is 'ParEGO':
        sampling_policy = ParEGO(model, space, utility)

    # BO model
    max_iter = 100
    experiment_name = 'test_GP1'
    if len(sys.argv) > 1:
        experiment_number = str(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, experiment_number]

        # True underlying utility
        random_state = np.random.RandomState(int(sys.argv[1]))
        true_underlying_utility_parameter = np.empty((2, ))
        alpha = 0.5 * np.pi * random_state.rand(1)
        true_underlying_utility_parameter[0] = np.cos(alpha)
        true_underlying_utility_parameter[1] = np.sin(alpha)
        print(true_underlying_utility_parameter)

        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)

        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=False)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
    else:
        for i in range(1):
            experiment_number = str(i)
            filename = [experiment_name, sampling_policy_name, experiment_number]

            # True underlying utility
            random_state = np.random.RandomState(int(sys.argv[1]))
            true_underlying_utility_parameter = np.empty((2,))
            alpha = 0.5 * np.pi * random_state.rand(1)
            true_underlying_utility_parameter[0] = np.cos(alpha)
            true_underlying_utility_parameter[1] = np.sin(alpha)
            print(true_underlying_utility_parameter)

            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)

            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=False)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)

