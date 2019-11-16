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
    from sampling_policies.acquisition_functions import uEI_constrained
    from utility import UtilityDistribution
    from utility import Utility
    from utility import ExpectationUtility
    from utility.elicitation_strategies import random_preference_elicitation
    from bopu import BOPU
    from optimization_services import U_AcquisitionOptimizer
    # Required for portfolio simulation
    import warnings
    import pandas as pd
    import cvxportfolio as cp

    # Set up requirements for portfolio simulation
    datadir = script_dir + '/portfolio_test_data/'

    sigmas = pd.read_csv(datadir + 'sigmas.csv.gz', index_col=0, parse_dates=[0]).iloc[:, :-1]
    returns = pd.read_csv(datadir + 'returns.csv.gz', index_col=0, parse_dates=[0])
    volumes = pd.read_csv(datadir + 'volumes.csv.gz', index_col=0, parse_dates=[0]).iloc[:, :-1]

    w_b = pd.Series(index=returns.columns, data=1)
    w_b.USDOLLAR = 0.
    w_b /= sum(w_b)

    start_t = "2012-01-01"
    end_t = "2016-12-31"

    simulated_tcost = cp.TcostModel(half_spread=0.0005 / 2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
    simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
    simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                                   market_volumes=volumes, cash_key='USDOLLAR')

    return_estimate = pd.read_csv(datadir + 'return_estimate.csv.gz', index_col=0, parse_dates=[0]).dropna()
    volume_estimate = pd.read_csv(datadir + 'volume_estimate.csv.gz', index_col=0, parse_dates=[0]).dropna()
    sigma_estimate = pd.read_csv(datadir + 'sigma_estimate.csv.gz', index_col=0, parse_dates=[0]).dropna()

    optimization_tcost = cp.TcostModel(half_spread=0.0005 / 2., nonlin_coeff=1.,
                                       sigma=sigma_estimate, volume=volume_estimate)
    optimization_hcost = cp.HcostModel(borrow_costs=0.0001)

    # Space
    d = 3
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Model (Multi-output GP)
    m = 2
    model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)
    # model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

    # Utility function
    def utility_func(y, parameter):
        y_aux = np.squeeze(y)
        parameter_aux = np.squeeze(parameter)
        if y_aux.ndim > 1:
            val = np.empty((y_aux.shape[1], ))
            for i in range(y_aux.shape[1]):
                if parameter_aux < y_aux[1, i]:
                    val[i] = y_aux[0, i]
                else:
                    val[i] = -np.infty
        else:
            if parameter_aux < y_aux[1]:
                val = y_aux[0]
            else:
                val = -np.infty
        return val

    def prior_sample_generator(n_samples, seed=None):
        if seed is None:
            samples = np.random.rand(n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.rand(n_samples)
        samples = 8.0 * samples + 2.0
        samples = -np.reshape(samples, (n_samples, 1))
        return samples

    utility_parameter_distribution = UtilityDistribution(prior_sample_generator=prior_sample_generator, utility_func=utility_func, elicitation_strategy=random_preference_elicitation)
    utility = Utility(func=utility_func, parameter_distribution=utility_parameter_distribution,
                      affine=False)

    # --- Sampling policy
    sampling_policy_name = 'uTS'
    if sampling_policy_name is 'uEI':
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='CMA', n_starting=80, n_anchor=8, include_baseline_points=False)
        acquisition = uEI_constrained(model, space, optimizer=acquisition_optimizer, utility=utility)
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
    max_iter = 100
    experiment_name = 'test_portfolio2'
    if len(sys.argv) > 1:
        experiment_number = int(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, str(experiment_number)]

        # Attributes
        copy_of_risk_model_name = 'risk_model_' + filename[0] + '_' + filename[1] + '_' + filename[2] + '.h5'
        subprocess.call(['bash', script_dir + '/make_copy_of_risk_model.sh', datadir + 'risk_model.h5', datadir + copy_of_risk_model_name])
        risk_data = pd.HDFStore(datadir + copy_of_risk_model_name)
        risk_model = cp.FactorModelSigma(risk_data.exposures, risk_data.factor_sigma, risk_data.idyos)

        def f_unnormalized_inputs(gamma_risk, gamma_tcost, gamma_holding):
            fx = np.empty((m,))
            results = {}
            policies = {}
            policies[(gamma_risk, gamma_tcost, gamma_holding)] = cp.SinglePeriodOpt(return_estimate,
                                                                                    [gamma_risk * risk_model,
                                                                                     gamma_tcost * optimization_tcost,
                                                                                     gamma_holding * optimization_hcost],
                                                                                    [cp.LeverageLimit(3)])
            warnings.filterwarnings('ignore')
            results.update(dict(zip(policies.keys(),
                                    simulator.run_multiple_backtest(1E8 * w_b, start_time=start_t, end_time=end_t,
                                                                    policies=policies.values(), parallel=True))))
            results_df = pd.DataFrame()
            results_df[r'$\gamma^\mathrm{risk}$'] = [el[0] for el in results.keys()]
            results_df[r'$\gamma^\mathrm{trade}$'] = [el[1] for el in results.keys()]
            results_df[r'$\gamma^\mathrm{hold}$'] = ['%g' % el[2] for el in results.keys()]
            results_df['Return'] = [results[k].excess_returns for k in results.keys()]
            for k in results.keys():
                returns = results[k].excess_returns.to_numpy()
            returns = returns[:-1]
            fx[0] = np.mean(returns) * 100 * 250
            fx[1] = -np.std(returns) * 100 * np.sqrt(250)
            return fx

        def f(X):
            X = np.atleast_2d(X)
            fX = np.empty((m, X.shape[0]))
            for i in range(X.shape[0]):
                gamma_risk = 999.9 * X[i, 0] + 0.1
                gamma_tcost = 2.5 * X[i, 1] + 5.5
                gamma_holding = 99.9 * X[i, 2] + 0.1
                fX[:, i] = f_unnormalized_inputs(gamma_risk, gamma_tcost, gamma_holding)
            print(fX)
            return fX

        attributes = Attributes(f, as_list=False, output_dim=m)

        # Initial design
        initial_design = GPyOpt.experiment_design.initial_design('random', space, 1, experiment_number)
        feasible_point = np.atleast_1d([0.31, 0.27, 1.])
        initial_design = np.vstack((initial_design, feasible_point))

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
        subprocess.call(['bash', script_dir + '/delete_copy_of_risk_model.sh', datadir + copy_of_risk_model_name])
    else:
        for i in range(1):
            experiment_number = i
            filename = [experiment_name, sampling_policy_name, str(experiment_number)]

            # Attributes
            copy_of_risk_model_name = 'risk_model_' + filename[0] + '_' + filename[1] + '_' + filename[2] + '.h5'
            subprocess.call(['bash', script_dir + '/make_copy_of_risk_model.sh', datadir + 'risk_model.h5',
                             datadir + copy_of_risk_model_name])
            risk_data = pd.HDFStore(datadir + copy_of_risk_model_name)
            risk_model = cp.FactorModelSigma(risk_data.exposures, risk_data.factor_sigma, risk_data.idyos)

            def f_unnormalized_inputs(gamma_risk, gamma_tcost, gamma_holding):
                fx = np.empty((m,))
                results = {}
                policies = {}
                policies[(gamma_risk, gamma_tcost, gamma_holding)] = cp.SinglePeriodOpt(return_estimate,
                                                                                        [gamma_risk * risk_model,
                                                                                         gamma_tcost * optimization_tcost,
                                                                                         gamma_holding * optimization_hcost],
                                                                                        [cp.LeverageLimit(3)])
                warnings.filterwarnings('ignore')
                results.update(dict(zip(policies.keys(),
                                        simulator.run_multiple_backtest(1E8 * w_b, start_time=start_t, end_time=end_t,
                                                                        policies=policies.values(), parallel=True))))
                results_df = pd.DataFrame()
                results_df[r'$\gamma^\mathrm{risk}$'] = [el[0] for el in results.keys()]
                results_df[r'$\gamma^\mathrm{trade}$'] = [el[1] for el in results.keys()]
                results_df[r'$\gamma^\mathrm{hold}$'] = ['%g' % el[2] for el in results.keys()]
                results_df['Return'] = [results[k].excess_returns for k in results.keys()]
                for k in results.keys():
                    returns = results[k].excess_returns.to_numpy()
                returns = returns[:-1]
                fx[0] = np.mean(returns) * 100 * 250
                fx[1] = -np.std(returns) * 100 * np.sqrt(250)
                return fx

            def f(X):
                X = np.atleast_2d(X)
                fX = np.empty((m, X.shape[0]))
                for i in range(X.shape[0]):
                    gamma_risk = 999.9 * X[i, 0] + 0.1
                    gamma_tcost = 2.5 * X[i, 1] + 5.5
                    gamma_holding = 99.9 * X[i, 2] + 0.1
                    fX[:, i] = f_unnormalized_inputs(gamma_risk, gamma_tcost, gamma_holding)
                print(fX)
                return fX

            attributes = Attributes(f, as_list=False, output_dim=m)

            # Initial design
            initial_design = GPyOpt.experiment_design.initial_design('random', space, 2 * (d + 1), experiment_number)
            feasible_point = np.atleast_1d([0.31, 0.27, 1.])
            initial_design = np.vstack((initial_design, feasible_point))

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
            subprocess.call(['bash', script_dir + '/delete_copy_of_risk_model.sh', datadir + copy_of_risk_model_name])
