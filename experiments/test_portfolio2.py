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

    # Function to optimize
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

    risk_data = pd.HDFStore(datadir + 'risk_model4.h5')
    risk_model = cp.FactorModelSigma(risk_data.exposures, risk_data.factor_sigma, risk_data.idyos)

    d = 3
    m = 2

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
        fx[0] = np.mean(returns)*100*250
        fx[1] = -np.std(returns)*100*np.sqrt(250)
        return fx

    def f(X):
        X = np.atleast_2d(X)
        fX = np.empty((m, X.shape[0]))
        for i in range(X.shape[0]):
            gamma_risk = 9.0 * X[i, 0] + 1.0
            gamma_tcost = 2.0 * X[i, 1] + 6.0
            gamma_holding = 990.0 * X[i, 2] + 10.0
            fX[:, i] = f_unnormalized_inputs(gamma_risk, gamma_tcost, gamma_holding)
        return fX

    attributes = Attributes(f, as_list=False, output_dim=m)

    # Function to optimize
    d = 3
    m = 2
    I = np.linspace(0., 1., 10)
    x, y, z = np.meshgrid(I, I, I)
    grid = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    kernel = GPy.kern.Matern52(input_dim=d, variance=2., ARD=True, lengthscale=np.atleast_1d([0.3] * d))
    cov = kernel.K(grid)
    mean = np.zeros((1000,))
    r1 = np.random.RandomState(1)
    Y1 = r1.multivariate_normal(mean, cov)
    r2 = np.random.RandomState(2)
    Y2 = r2.multivariate_normal(mean, cov)
    Y1 = np.reshape(Y1, (1000, 1))
    Y2 = np.reshape(Y2, (1000, 1))
    # print(Y1[:5, 0])
    # print(Y2[:5, 0])
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

    # Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

    # Model (Multi-output GP)
    model = MultiOutputGP(output_dim=m, exact_feval=[True] * m, fixed_hyps=False)
    # model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

    # Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 2*(d+1))

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

    def prior_sample_generator(n_samples):
        samples = 8.0 * np.random.rand(n_samples) + 2.0
        samples = -np.reshape(samples, (n_samples, 1))
        return samples

    utility_parameter_distribution = UtilityDistribution(prior_sample_generator=prior_sample_generator, utility_func=utility_func, elicitation_strategy=random_preference_elicitation)
    utility = Utility(func=utility_func, parameter_distribution=utility_parameter_distribution,
                      affine=False)

    # --- Sampling policy
    sampling_policy_name = 'uEI'
    if sampling_policy_name is 'uEI':
        # Acquisition optimizer
        acquisition_optimizer = U_AcquisitionOptimizer(space=space, model=model, utility=utility, optimizer='CMA', n_anchor=4)

        acquisition = uEI_constrained(model, space, optimizer=acquisition_optimizer, utility=utility)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        sampling_policy = AcquisitionFunction(model, space, acquisition, evaluator)
    elif sampling_policy_name is 'TS':
        sampling_policy = TS(model, optimization_space, optimizer='CMA', scenario_distribution=scenario_distribution,
                             utility=utility, expectation_utility=expectation_utility)
        acquisition = None
    elif sampling_policy_name is 'Random':
        sampling_policy = Random(model, space)

    # BO model
    max_iter = 1
    experiment_name = 'test_portfolio2'
    if len(sys.argv) > 1:
        experiment_number = str(sys.argv[1])
        filename = [experiment_name, sampling_policy_name, experiment_number]

        # True underlying utility
        true_underlying_utility_parameter = utility.sample_parameter()[0]
        print(true_underlying_utility_parameter)

        def true_underlying_utility_func(y):
            return utility_func(y, true_underlying_utility_parameter)

        bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                    true_underlying_utility_func=true_underlying_utility_func,
                    dynamic_utility_parameter_distribution=True)
        bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                              utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                              compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)
    else:
        for i in range(1):
            experiment_number = str(i)
            filename = [experiment_name, sampling_policy_name, experiment_number]

            # True underlying utility
            true_underlying_utility_parameter = utility.sample_parameter()[0]
            print(true_underlying_utility_parameter)

            def true_underlying_utility_func(y):
                return utility_func(y, true_underlying_utility_parameter)

            bopu = BOPU(model, space, attributes, sampling_policy, utility, initial_design,
                        true_underlying_utility_func=true_underlying_utility_func,
                        dynamic_utility_parameter_distribution=True)
            bopu.run_optimization(max_iter=max_iter, filename=filename, report_evaluated_designs_only=True,
                                  utility_distribution_update_interval=1, compute_true_underlying_optimal_value=False,
                                  compute_integrated_optimal_values=True, compute_true_integrated_optimal_value=True)