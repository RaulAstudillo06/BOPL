if __name__ == '__main__':
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys

    sys.path.append(script_dir[:-11] + 'bopu')
    import numpy as np
    import aux_software.GPyOpt as GPyOpt
    import aux_software.GPy as GPy
    from core import Attributes
    from models import MultiOutputGP
    from models import BasicModel
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

    # Input and output dimensions
    m = 4
    k = 2
    d = m + k - 1

    # Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Attributes
    def f(X):
        gX = np.zeros((X.shape[0], ))
        fX = np.ones((m, X.shape[0]))
        for i in range(k):
            gX += np.square(X[:, d - i - 1] - 0.5)
        for i in range(m - 1):
            fX[0, :] *= np.cos(0.5 * np.pi * X[:, i])
        for i in range(m - 2):
            fX[1, :] *= np.cos(0.5 * np.pi * X[:, i])
        fX[1, :] *= np.sin(0.5 * np.pi * X[:, m - 2])
        fX[2, :] = np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1])
        fX[3, :] = np.sin(0.5 * np.pi * X[:, 0])
        for j in range(m):
            fX[j, :] *= (1 + gX)
        return -fX

    attributes = Attributes(f, as_list=False, output_dim=m)

    # Utility function
    def utility_func(y, parameter):
        aux = (y.transpose() - parameter).transpose()
        return -np.sum(np.square(aux), axis=0)

    def utility_gradient(y, parameter):
        y_aux = np.squeeze(y)
        return -2 * (y_aux - parameter)

        #  Parameter distribution
    X1 = [0., 1./3.]
    X2 = [1./3., 2./3.]
    X3 = [2./3., 1.]
    X4 = [0.5]
    X5 = [0.5]
    grid = np.meshgrid(X1, X2, X3, X4, X5)
    X_pareto = np.array([a.flatten() for a in grid]).T
    utility_parameter_support = f(X_pareto).T
    utility_parameter_prob_dist = np.ones((8,)) / 8.
    utility_parameter_distribution = UtilityDistribution(support=utility_parameter_support,
                                                     prob_dist=utility_parameter_prob_dist,
                                                     utility_func=utility_func,
                                                     elicitation_strategy=random_preference_elicitation)

        # Expectation of utility
    def expectation_utility_func(mu, var, parameter):
        aux = (mu.transpose() - parameter).transpose()
        val = -np.sum(np.square(aux), axis=0) - np.sum(var, axis=0)
        return val

    def expectation_utility_gradient(mu, var, parameter):
        mu_aux = np.squeeze(mu)
        var_aux = np.squeeze(var)
        gradient = -np.concatenate((2 * (mu_aux - parameter), np.ones((len(var_aux),))))
        return gradient

    expectation_utility = ExpectationUtility(expectation_utility_func, expectation_utility_gradient)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_parameter_distribution,
                      expectation=expectation_utility)

    # --- Sampling policy
    sampling_policy_name = 'uEI'
    learn_preferences = True
    if sampling_policy_name is 'uEI':
        model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)  # Model (Multi-output GP)
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='lbfgs',
                                                       include_baseline_points=True)
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
    experiment_name = 'test_dtlz2a'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]

        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

        # True underlying utility
        true_underlying_utility_parameter = utility.sample_parameter_from_prior(1, experiment_number)
        print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))


        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)


        # Run full optimization loop
        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                    true_underlying_utility_func=true_underlying_utility_func,
                    dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                              utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True,
                              compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)
    else:
        for i in range(1):
            experiment_number = i
            filename = [experiment_name, sampling_policy_name, str(experiment_number)]

            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

            # True underlying utility
            true_underlying_utility_parameter = utility.sample_parameter_from_prior(1, experiment_number)
            print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))


            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)


            # Run full optimization loop
            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                        true_underlying_utility_func=true_underlying_utility_func,
                        dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                                  utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True,
                                  compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)