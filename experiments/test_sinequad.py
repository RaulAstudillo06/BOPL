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
    from utility import UtilityDistribution
    from utility import Utility
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer

    # Space
    d = 2
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Model (Multi-output GP)
    m = 1
    model = multi_outputGP(output_dim=m, fixed_hyps=False)

    # Utility function
    def aux_utility_func(parameter, y):
        return np.dot(parameter, y)


    def aux_utility_gradient(parameter, y):
        return parameter


    utility_parameter_support = np.ones((1, 1))
    utility_parameter_prob_distribution = np.ones((1,))
    utility_param_distribution = UtilityDistribution(support=utility_parameter_support,
                                                     prob_dist=utility_parameter_prob_distribution,
                                                     utility_func=utility_func)
    dynamic_utility_parameter_distribution = False
    utility = Utility(func=aux_utility_func, gradient=aux_utility_gradient,
                      parameter_distribution=aux_utility_param_distribution, affine=True)

    # --- Sampling policy
    sampling_policy_name = 'Random'
    if sampling_policy_name is 'uTS':
        model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)  # Model (Multi-output GP)
        sampling_policy = uTS(model, space, optimizer='CMA', utility=utility)
    elif sampling_policy_name is 'Random':
        model = BasicModel(output_dim=m)
        sampling_policy = Random(model=None, space=space)

    # Attributes
    def f(X):
        X = np.atleast_2d(X)
        fX = np.sin(10 * X[:, 0]) + np.square(X[:, 1])
        fX = np.atleast_2d(fX)
        return fX

    attributes = Attributes(f, noise_var=[0.01], as_list=False, output_dim=m)

    # BO model
    max_iter = 150
    experiment_name = 'test_sinequad'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]

        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)

        # Run full optimization loop
        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                    true_underlying_utility_func=true_underlying_utility_func,
                    dynamic_utility_parameter_distribution=dynamic_utility_parameter_distribution)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=False,
                              utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                              compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)