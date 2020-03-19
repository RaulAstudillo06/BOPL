if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(script_dir[:-11] + 'bopl')
    import numpy as np
    import subprocess
    from core import Attributes
    from core import BOPL
    from models import BasicModel
    from models import MultiOutputGP
    from sampling_policies import Random
    from sampling_policies import uTS
    from utility import Utility
    from utility import UtilityDistribution
    from optimization_services import AcquisitionOptimizer
    import aux_software.GPyOpt as GPyOpt

    # Space
    d = 2
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Model (Multi-output GP)
    m = 1
    model = MultiOutputGP(output_dim=m, fixed_hyps=False)

    # Utility function
    def utility_func(y, parameter):
        return np.squeeze(y)


    def utility_gradient(y, parameter):
        return parameter


    utility_parameter_support = np.ones((1, 1))
    utility_parameter_prob_distribution = np.ones((1,))
    utility_param_distribution = UtilityDistribution(support=utility_parameter_support,
                                                     prob_dist=utility_parameter_prob_distribution,
                                                     utility_func=utility_func)
    dynamic_utility_parameter_distribution = False
    utility = Utility(func=utility_func, gradient=utility_gradient,
                      parameter_distribution=utility_param_distribution, affine=True)

    # --- Sampling policy
    sampling_policy_name = 'Random'
    if sampling_policy_name is 'uTS':
        sampling_policy = uTS(model, space, optimizer='CMA', utility=utility)
    elif sampling_policy_name is 'Random':
        sampling_policy = Random(model, space)

    # Attributes
    def f(X):
        X = np.atleast_2d(X)
        fX = -np.sin(10 * X[:, 0]) - np.square(X[:, 1])
        fX = np.atleast_2d(fX)
        return fX

    attributes = Attributes(f, noise_var=[0.01], as_list=False, output_dim=m)

    # BO model
    max_iter = 100
    experiment_name = 'test_sinequad'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]

        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

        # Run full optimization loop
        bopl = BOPL(model, space, attributes, sampling_policy, utility, initial_design, dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
        bopl.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=False,
                              utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                              compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)
    else:
        for i in range(1):
            experiment_number = i
            filename = [experiment_name, sampling_policy_name, str(experiment_number)]
            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)
    
            # Run full optimization loop
            bopl = BOPL(model, space, attributes, sampling_policy, utility, initial_design, dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
            bopl.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=False,
                                  utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                                  compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)
            