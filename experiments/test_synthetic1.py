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
    from sampling_policies.acquisition_functions import uEI_affine
    from utility import UtilityDistribution
    from utility import Utility
    from utility import ExpectationUtility
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer
    
    # Space
    d = 4
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (-2, 2), 'dimensionality': d}])
    
    # Model (Multi-output GP)
    m = 3
    model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)
    
    # Attributes
    # Rosenbrock function(-X)
    def f1(X_minus):
        X = np.atleast_2d(-X_minus)
        fX = np.zeros((X.shape[0], 1))
        for j in range(d - 1):
            fX[:, 0] += 100*((X[:, j + 1] - X[:, j]**2)**2) + (X[:, j] - 1)**2
        return -fX

    # Ackley function
    def f2(X):
        a = 20.
        b = 0.2
        c = 2*np.pi
        X = np.atleast_2d(X)
        fX = np.zeros((X.shape[0], 1))
        fX[:, 0] = -a*np.exp(-b*np.sqrt((1./d)*np.sum(np.square(X), axis=1))) - np.exp((1./d)*np.sum(np.cos(c*X), axis=1)) + a + np.exp(1.)
        return -fX

    # Levy function
    def f3(X):
        W = 1. + (np.atleast_2d(X) -1.)/4.
        fX = np.zeros((W.shape[0], 1))
        fX[:, 0] = np.square(np.sin(np.pi*W[:,0])) + np.square(W[:, d-1] - 1.)*(1. + np.square(np.sin(2*np.pi*W[:,0])))
        for i in range(d-1):
            fX[:, 0] += np.square(W[:, i] - 1)*(1. + 10.*np.square(np.sin(np.pi*W[:, i] + 1.)))
        return -fX
     
    attributes = Attributes([f1, f2, f3], as_list=True, output_dim=m)
    
    # Utility function
    def utility_func(y, parameter):
        return np.dot(parameter, y)


    def utility_gradient(y, parameter):
        return parameter
    
        # Parameter distribution
    L = 3
    support = np.eye(L)
    prob_dist = np.ones((L,)) / L
    
    utility_parameter_distribution = UtilityDistribution(support=support, prob_dist=prob_dist, utility_func=utility_func, elicitation_strategy=None)
    utility = Utility(func=utility_func, gradient=utility_gradient, parameter_distribution=utility_parameter_distribution, affine=True)
    
    # --- Sampling policy
    sampling_policy_name = 'uEI'
    if sampling_policy_name is 'uEI':
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='lbfgs', include_baseline_points=True)
        acquisition = uEI_affine(model, space, optimizer=acquisition_optimizer, utility=utility)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        sampling_policy = AcquisitionFunction(model, space, acquisition, evaluator)
    elif sampling_policy_name is 'uTS':
        model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)  # Model (Multi-output GP)
        sampling_policy = uTS(model, space, optimizer='CMA', utility=utility)
    elif sampling_policy_name is 'Random':
        model = BasicModel(output_dim=m)
        sampling_policy = Random(model=None, space=space)
    elif sampling_policy_name is 'ParEGO':
        model = BasicModel(output_dim=m)
        sampling_policy = ParEGO(model, space, utility)
        
    # BO model
    max_iter = 100
    experiment_name = 'test_synthetic1'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]
    
        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)
    
        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design, dynamic_utility_parameter_distribution=False)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                              utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                              compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)
    else:
        for i in range(1):
            experiment_number = i
            filename = [experiment_name, sampling_policy_name, str(experiment_number)]
    
            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)
    
            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design, dynamic_utility_parameter_distribution=False)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                                  utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                                  compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)