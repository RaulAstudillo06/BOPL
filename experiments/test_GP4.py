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
    d = 4
    m = 4

    # Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Attributes (preliminaries)
    I = np.linspace(0., 1., 6)
    aux = np.meshgrid(I, I, I, I)
    grid = np.array([a.flatten() for a in aux]).T
    kernel = []
    kernel.append(GPy.kern.Matern52(input_dim=d, variance=2., ARD=True, lengthscale=np.atleast_1d([0.1, 0.2, 0.3, 0.4])))
    kernel.append(GPy.kern.Matern52(input_dim=d, variance=2., ARD=True, lengthscale=np.atleast_1d([0.3, 0.4, 0.5, 0.6])))
    cov = []
    cov.append(kernel[0].K(grid))
    cov.append(kernel[1].K(grid))
    mean = np.zeros((6 ** d,))

    # Utility function
    def utility_func(y, theta):
        y_aux = np.squeeze(y)
        val = np.sum(1. - np.exp(-theta*y_aux), axis=0)/theta
        return val

    def utility_gradient(y, theta):
        y_aux = np.squeeze(y)
        gradient = np.exp(-theta*y_aux)
        return gradient

        #  Parameter distribution
    def prior_sample_generator(n_samples, seed=None):
        if seed is None:
            samples = np.random.rand(n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.rand(n_samples)
        samples = 0.4*samples + 0.1
        return samples

    utility_parameter_distribution = UtilityDistribution(prior_sample_generator=prior_sample_generator,
                                                     utility_func=utility_func,
                                                     elicitation_strategy=random_preference_elicitation)

        # Expectation of utility
    def expectation_utility_func(mean, var, theta):
        mean_aux = np.squeeze(mean)
        var_aux = np.squeeze(var)
        val = np.sum(1. - np.exp(-theta*mean_aux + np.square(theta)*var_aux), axis=0)/theta
        return val

    def expectation_utility_gradient(mean, var, theta):
        mean_aux = np.squeeze(mean)
        var_aux = np.squeeze(var)
        mean_gradient = np.exp(-theta*mean_aux + 0.5*np.square(theta)*var_aux)
        var_gradient = -0.5*theta*mean_gradient
        gradient = -np.concatenate((mean_gradient, var_gradient))
        return gradient

    expectation_utility = ExpectationUtility(expectation_utility_func, expectation_utility_gradient)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_parameter_distribution,
                      expectation=expectation_utility)

    # --- Sampling policy
    sampling_policy_name = 'Random'
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
    max_iter = 1
    experiment_name = 'test_GP4'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]

        # Attributes
        aux_model = []
        for j in range(m):
            r = np.random.RandomState(j + experiment_number)
            Y = r.multivariate_normal(mean, cov[j % 2])
            Y = np.reshape(Y, (6 ** d, 1))
            aux_model.append(GPy.models.GPRegression(grid, Y, kernel[j % 2], noise_var=1e-10))

        def f(X):
            X = np.atleast_2d(X)
            fX = np.empty((m, X.shape[0]))
            for j in range(m):
                fX[j, :] = aux_model[j].posterior_mean(X)[:, 0]
            return fX

        attributes = Attributes(f, as_list=False, output_dim=m)

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
                              compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
    else:
        for i in range(1):
            experiment_number = i
            filename = [experiment_name, sampling_policy_name, str(experiment_number)]

            # Attributes
            aux_model = []
            for j in range(m):
                r = np.random.RandomState(j + experiment_number)
                Y = r.multivariate_normal(mean, cov[j % 2])
                Y = np.reshape(Y, (6 ** d, 1))
                aux_model.append(GPy.models.GPRegression(grid, Y, kernel[j % 2], noise_var=1e-10))

            def f(X):
                X = np.atleast_2d(X)
                fX = np.empty((m, X.shape[0]))
                for j in range(m):
                    fX[j, :] = aux_model[j].posterior_mean(X)[:, 0]
                return fX

            attributes = Attributes(f, as_list=False, output_dim=m)

            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

            # True underlying utility
            true_underlying_utility_parameter = 0.1 #utility.sample_parameter_from_prior(1, experiment_number)
            print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))


            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)


            # Run full optimization loop
            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                        true_underlying_utility_func=true_underlying_utility_func,
                        dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                                  utility_distribution_update_interval=1, compute_true_underlying_optimal_value=True,
                                  compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)