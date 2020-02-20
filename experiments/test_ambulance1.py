if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(script_dir[:-11] + 'bopu')
    import numpy as np
    import subprocess
    import aux_software.GPyOpt as GPyOpt
    import aux_software.GPy as GPy
    from core import Attributes
    from models import MultiOutputGP
    from models import BasicModel
    from sampling_policies import Random
    from sampling_policies import uTS
    from sampling_policies import ParEGO
    from sampling_policies import AcquisitionFunction
    from sampling_policies.acquisition_functions import uEI
    from utility import UtilityDistribution
    from utility import Utility
    from utility import ExpectationUtility
    from utility.elicitation_strategies import random_preference_elicitation
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer
    from ambulance_simulation import ambulance_simulation
    import pandas as pd
    
    # Space
    d = 6
    n_ambulances = int(d/2)
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])
    
    # Model (Multi-output GP)
    m = 5
    model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)
    
    # Utility function
    def utility_func(y, parameter):
        y_aux = np.squeeze(y)
        parameter_aux = np.squeeze(parameter)
        if y_aux.ndim > 1:
            val = np.empty((y_aux.shape[1], ))
            for i in range(y_aux.shape[1]):
                tmp = np.exp(y_aux[:, i])
                tmp /= np.sum(tmp)
                val[i] = np.dot(parameter, tmp)
        else:
            tmp = np.exp(y_aux)
            tmp /= np.sum(tmp)
            val = np.dot(parameter, tmp)
        return val
    
    def utility_gradient(y, parameter):
        y_aux = np.squeeze(y)
        parameter_aux = np.squeeze(parameter)
        tmp1 = np.exp(y_aux)
        tmp2 = np.multiply(parameter, tmp1)
        s1 = np.sum(tmp1)
        s2 = np.sum(tmp2)
        gradient = np.empty((m, )) 
        for j in range(m):
            gradient[j] = s1*tmp2[j] - s2*tmp1[j]         
        gradient /= np.square(s1)
        return gradient
    
    def prior_sample_generator(n_samples=1, seed=None):
        if seed is None:
            samples = np.random.dirichlet(np.ones((m, )), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((m, )), n_samples)
        samples = -np.sort(-samples)
        return samples
    
    utility_parameter_distribution = UtilityDistribution(prior_sample_generator=prior_sample_generator, utility_func=utility_func, elicitation_strategy=random_preference_elicitation)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_parameter_distribution,
                      affine=False)
    
    # --- Sampling policy
    sampling_policy_name = 'uTS'
    if sampling_policy_name is 'uEI':
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='lbfgs', include_baseline_points=True)
        acquisition = uEI(model, space, optimizer=acquisition_optimizer, utility=utility)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        sampling_policy = AcquisitionFunction(model, space, acquisition, evaluator)
        dynamic_utility_parameter_distribution = True
    elif sampling_policy_name is 'uTS':
        model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)  # Model (Multi-output GP)
        sampling_policy = uTS(model, space, optimizer='CMA', utility=utility)
        dynamic_utility_parameter_distribution = True
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
    experiment_name = 'test_ambulance1'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]
    
        # Attributes
        def f(X):
            X = np.atleast_2d(X)
            fX = np.zeros((m, X.shape[0]))
            base_loc_x = np.empty((n_ambulances, ))
            base_loc_y = np.empty((n_ambulances, ))
            for i in range(X.shape[0]):
                for j in range(n_ambulances):
                    base_loc_x[j] = X[i, j]
                    base_loc_y[j] = X[i, j + n_ambulances]
                response_times = ambulance_simulation(base_loc_x, base_loc_y, seed=experiment_number)
                response_times = 60*response_times
                for t in range(len(response_times)):
                    if response_times[t] <= 5:
                        fX[0, i] += 1
                    elif 5 < response_times[t] and response_times[t] <= 10:
                        fX[1, i] += 1
                    elif 10 < response_times[t] and response_times[t] <= 15:
                        fX[2, i] += 1
                    elif 15 < response_times[t] and response_times[t] <= 20:
                        fX[3, i] += 1
                    else:
                        fX[4, i] += 1
            fX = np.log(fX)
            print(fX)
            return fX

        attributes = Attributes(f, as_list=False, output_dim=m)
    
        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)
    
        # True underlying utility
        true_underlying_utility_parameter = prior_sample_generator(1, experiment_number)[0]
        print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))
    
        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)
    
        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                    true_underlying_utility_func=true_underlying_utility_func,
                    dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                              utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                              compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
    else:
        for i in range(1):
            experiment_number = i
            filename = [experiment_name, sampling_policy_name, str(experiment_number)]
    
            # Attributes
            def f(X):
                X = np.atleast_2d(X)
                fX = np.zeros((m, X.shape[0]))
                base_loc_x = np.empty((n_ambulances, ))
                base_loc_y = np.empty((n_ambulances, ))
                for i in range(X.shape[0]):
                    for j in range(n_ambulances):
                        base_loc_x[j] = X[i, j]
                        base_loc_y[j] = X[i, j + n_ambulances]
                    response_times = ambulance_simulation(base_loc_x, base_loc_y, seed=experiment_number)
                    response_times = 60*response_times
                    for t in range(len(response_times)):
                        if response_times[t] <= 5:
                            fX[0, i] += 1
                        elif 5 < response_times[t] and response_times[t] <= 10:
                            fX[1, i] += 1
                        elif 10 < response_times[t] and response_times[t] <= 15:
                            fX[2, i] += 1
                        elif 15 < response_times[t] and response_times[t] <= 20:
                            fX[3, i] += 1
                        else:
                            fX[4, i] += 1
                fX = np.log(fX)
                print(fX)
                return fX
    
            attributes = Attributes(f, as_list=False, output_dim=m)
    
            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)
    
            # True underlying utility
            true_underlying_utility_parameter = prior_sample_generator(1, experiment_number)[0]
            print('True underlying utility parameter: {}'.format(true_underlying_utility_parameter))
    
            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)
    
            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                        true_underlying_utility_func=true_underlying_utility_func,
                        dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                                  utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                                  compute_integrated_optimal_values=False, compute_true_integrated_optimal_value=False)
