if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(script_dir[:-11] + 'bopl')
    import numpy as np
    from core import Attributes
    from core import BOPL
    from models import BasicModel
    from models import MultiOutputGP
    from sampling_policies import Random
    from sampling_policies import ParEGO
    from sampling_policies import uTS
    from sampling_policies import AcquisitionFunction
    from sampling_policies.acquisition_functions import uEI_affine
    from utility import Utility
    from utility import UtilityDistribution
    from utility.elicitation_strategies import random_preference_elicitation
    from optimization_services import AcquisitionOptimizer
    import aux_software.GPyOpt as GPyOpt

    # Input and output dimensions
    d = 6
    m = 2

    # Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Attributes
    def g(X):
        X = np.atleast_2d(X)
        gX = np.empty((X.shape[0], 1))
        gX[:, 0] = 100*(5 + np.sum((X[:, 1:]-0.5)**2, axis=1) - np.sum(np.cos(2*np.pi*(X[:, 1:] - 0.5)), axis=1))
        return gX

    def f1(X):
        X = np.atleast_2d(X)
        f1X = np.empty((X.shape[0], 1))
        f1X[:, 0] = 0.5*X[:, 0]*(1 + g(X)[:, 0])
        return -f1X

    def f2(X):
        X = np.atleast_2d(X)
        f2X = np.empty((X.shape[0], 1))
        f2X[:, 0] = 0.5*(1 - X[:, 0])*(1 + g(X)[:, 0])
        return -f2X
    
    attributes = Attributes([f1, f2], as_list=True, output_dim=m)
    
    # Utility function
    def utility_func(y, parameter):
        return np.dot(parameter, y)

    def utility_gradient(y, parameter):
        return parameter

    def prior_sample_generator(n_samples=1, seed=None):
        if seed is None:
            samples = np.random.dirichlet(np.ones((m, )), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((m, )), n_samples)
        return samples


    utility_param_distribution = UtilityDistribution(prior_sample_generator=prior_sample_generator,
                                                     utility_func=utility_func,
                                                     elicitation_strategy=random_preference_elicitation)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_param_distribution, affine=True)

    # --- Sampling policy
    sampling_policy_name = 'uEI'
    learn_preferences = False
    if sampling_policy_name is 'uEI':
        model = MultiOutputGP(output_dim=m, exact_feval=[True]*m, fixed_hyps=False) # Model (Multi-output GP)
        acquisition_optimizer = AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='lbfgs', include_baseline_points=True)
        acquisition = uEI_affine(model, space, optimizer=acquisition_optimizer, utility=utility)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        sampling_policy = AcquisitionFunction(model, space, acquisition, evaluator)
        if learn_preferences:
            dynamic_utility_parameter_distribution = True
        else:
            dynamic_utility_parameter_distribution = False
            sampling_policy_name = 'uEI_prior'
    elif sampling_policy_name is 'uTS':
        model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)  # Model (Multi-output GP)
        sampling_policy = uTS(model, space, optimizer='CMA', utility=utility)
        if learn_preferences:
            dynamic_utility_parameter_distribution = True
        else:
            dynamic_utility_parameter_distribution = False
            sampling_policy_name = 'uTS_prior'
    elif sampling_policy_name is 'Random':
        model = BasicModel(output_dim=m)
        sampling_policy = Random(model=None, space=space)
        dynamic_utility_parameter_distribution = False
    elif sampling_policy_name is 'ParEGO':
        model = BasicModel(output_dim=m)
        sampling_policy = ParEGO(model, space, utility)
        dynamic_utility_parameter_distribution = False

    # BO model
    max_iter = 150
    experiment_name = 'test_dtlz1a'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]

        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

        # True underlying utility
        true_underlying_utility_parameter = prior_sample_generator(1, experiment_number)[0]
        print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))

        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)
        
        # Run full optimization loop
        bopl = BOPL(model, space, attributes, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
        bopl.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
    else:
        for i in range(1):
            experiment_number = i
            filename = [experiment_name, sampling_policy_name, str(experiment_number)]

            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

            # True underlying utility
            true_underlying_utility_parameter = prior_sample_generator(1, experiment_number)[0]
            print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))

            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)

            # Run full optimization loop
            bopl = BOPL(model, space, attributes, sampling_policy, utility, initial_design, true_underlying_utility_func=true_underlying_utility_func, dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
            bopl.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True, utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True, compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)