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
    from sampling_policies.acquisition_functions import uEI
    from sampling_policies import uTS
    from sampling_policies import ParEGO
    from utility import UtilityDistribution
    from utility import Utility
    from utility import ExpectationUtility
    from utility.elicitation_strategies import random_preference_elicitation
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer

    # Function to optimize
    d = 6
    m = 4
    kernels = []
    aux_models = []
    X_init = np.loadtxt(script_dir + '/ambulances_test_data/X.txt')
    print(X_init.shape)
    Y_init = np.loadtxt(script_dir + '/ambulances_test_data/fX.txt')
    Y_init = np.transpose(Y_init)
    print(Y_init.shape)
    Y_init = np.log(Y_init/(1. - Y_init))
    Y_init = np.reshape(Y_init, (m, 1000, 1))
    kernels.append(GPy.kern.Matern52(input_dim=d, variance=1.23944185e-01,
                                       lengthscale=np.atleast_1d([6.01573523e-01, 5.01105665e-01, 2.55493969e-01, 6.57924795e-01, 7.97101925e-01, 4.91051642e-01]), ARD=True))
    kernels.append(GPy.kern.Matern52(input_dim=d, variance=1.85472255e-01,
                                     lengthscale=np.atleast_1d([5.74205950e-01, 7.91947011e-01, 5.89611488e-01, 5.06130952e-01, 4.40359135e-01, 4.24819685e-01]), ARD=True))
    kernels.append(GPy.kern.Matern52(input_dim=d, variance=1.37742218e-01,
                                     lengthscale=np.atleast_1d([5.42669750e-01, 8.25799758e-01, 3.73776044e-01, 3.66697667e-01, 5.64635746e-01, 8.36060593e-01]), ARD=True))
    kernels.append(GPy.kern.Matern52(input_dim=d, variance=9.96009310e-02,
                                     lengthscale=np.atleast_1d([4.18075267e-01, 2.14289855e-01, 9.12020615e-02, 4.08172521e-01, 2.01868455e+00, 9.60459928e-01]), ARD=True))
    for j in range(m):
        aux_models.append(GPy.models.GPRegression(X_init, Y_init[j, :, :], kernels[j], noise_var=1e-10))

    def f(X):
        X = np.atleast_2d(X)
        fX = np.empty((m, X.shape[0]))
        for j in range(m):
            fX[j, :] = aux_models[j].posterior_mean(X)[:, 0]
        return fX

    attributes = Attributes(f, as_list=False, output_dim=m)

    # Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Model (Multi-output GP)
    model = MultiOutputGP(output_dim=m, exact_feval=[True]*m, fixed_hyps=False)

    # Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1))

    # Utility function
    def utility_func(y, parameter):
        y_aux = np.atleast_1d(y)
        probability_matrix = np.exp(y_aux)
        probability_matrix = probability_matrix / (1. + probability_matrix)
        if y_aux.ndim == 1:
            probability_matrix = np.reshape(probability_matrix, (m, 1))
        probability_matrix = np.append(probability_matrix, np.atleast_2d(1. - np.sum(probability_matrix, axis=0)), 0)
        val = np.matmul(parameter, probability_matrix)
        return val

    def utility_gradient(y, parameter):
        y_aux = np.atleast_1d(y)
        derivative_matrix = np.exp(y_aux)
        derivative_matrix = derivative_matrix / np.square(1. + derivative_matrix)
        aux = parameter[:-1] - parameter[-1]
        gradient = np.multiply(aux, derivative_matrix)
        return gradient

    def prior_sample_generator(n_samples):
        samples = np.random.dirichlet(np.ones(m + 1), n_samples)
        samples = -np.sort(-samples, axis=0)
        return samples


    utility_param_distribution = UtilityDistribution(prior_sample_generator=prior_sample_generator,
                                                     utility_func=utility_func,
                                                     elicitation_strategy=random_preference_elicitation)
    # utility_param_distribution = UtilityDistribution(support=utility_parameter_support, prob_dist=utility_parameter_prob_distribution, utility_func=utility_func, elicitation_strategy=random_preference_elicitation)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_param_distribution,
                      affine=False)

    # --- Sampling policy
    sampling_policy_name = 'uEI'
    if sampling_policy_name is 'uEI':
        # Acquisition optimizer
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='lbfgs',
                                                       inner_optimizer='lbfgs', include_baseline_points=False)

        acquisition = uEI(model, space, optimizer=acquisition_optimizer, utility=utility)
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
    experiment_name = 'test_ambulance1'
    if len(sys.argv) > 1:
        experiment_number = str(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, experiment_number]

        # True underlying utility
        true_underlying_utility_parameter = np.random.dirichlet(np.ones(m + 1), )
        true_underlying_utility_parameter = -np.sort(-true_underlying_utility_parameter, axis=0)
        print(true_underlying_utility_parameter)

        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)


        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                    true_underlying_utility_func=true_underlying_utility_func,
                    dynamic_utility_parameter_distribution=True)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                              utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True,
                              compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
    else:
        for i in range(1):
            experiment_number = str(i)
            filename = [experiment_name, sampling_policy_name, experiment_number]

            # True underlying utility
            true_underlying_utility_parameter = np.random.dirichlet(np.ones(m + 1), )
            true_underlying_utility_parameter = -np.sort(-true_underlying_utility_parameter, axis=0)
            print(true_underlying_utility_parameter)

            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)

            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                        true_underlying_utility_func=true_underlying_utility_func,
                        dynamic_utility_parameter_distribution=True)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                                  utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True,
                                  compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
