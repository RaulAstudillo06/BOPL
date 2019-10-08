from aux_software.GPyOpt.experiment_design import initial_design
from aux_software.GPyOpt.optimization.optimizer import apply_optimizer, choose_optimizer, apply_optimizer_inner
from aux_software.GPyOpt.optimization.anchor_points_generator import ObjectiveAnchorPointsGenerator, ThompsonSamplingAnchorPointsGenerator
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np

max_objective_anchor_points_logic = "max_objective"
thompson_sampling_anchor_points_logic = "thompsom_sampling"
sobol_design_type = "sobol"
random_design_type = "random"
latin_design_type = "latin"


class U_AcquisitionOptimizer(object):
    """
    General class for acquisition optimizers defined in domains with mix of discrete, continuous, bandit variables

    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    """

    def __init__(self, space, model, utility, expectation_utility=None, optimizer='lbfgs', inner_optimizer='lbfgs', parallel=False, n_starting=360, n_anchor=16, include_baseline_points=True, **kwargs):

        self.space = space
        self.model = model
        self.utility = utility
        self.expectation_utility = expectation_utility
        self.optimizer_name = optimizer
        self.inner_optimizer_name = inner_optimizer
        self.parallel = parallel
        self.n_starting = n_starting
        self.n_anchor = n_anchor
        self.include_baseline_points = include_baseline_points
        self.number_of_utility_parameter_samples = 10
        self.full_parameter_support = self.utility.parameter_distribution.use_full_support
        self.number_of_gp_hyps_samples = min(10, self.model.number_of_hyps_samples())
        self.kwargs = kwargs

        ## -- save extra options than can be passed to the optimizer
        if 'model' in self.kwargs:
            self.model = self.kwargs['model']

        if 'anchor_points_logic' in self.kwargs:
            self.type_anchor_points_logic = self.kwargs['type_anchor_points_logic']
        else:
            self.type_anchor_points_logic = max_objective_anchor_points_logic

        ## -- Context handler: takes
        self.context_manager = ContextManager(space)
        ## -- Set optimizer and inner optimizer (WARNING: this won't update context)
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)
        self.inner_optimizer = choose_optimizer(self.inner_optimizer_name, self.context_manager.noncontext_bounds)
    
    
    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df
        

        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, self.n_starting)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        # Select the anchor points (with context)
        anchor_points, anchor_points_values = anchor_points_generator.get(num_anchor=self.n_anchor, duplicate_manager=duplicate_manager, context_manager=self.context_manager, get_scores=True)
        # Baseline points
        if self.include_baseline_points:
            X_baseline = []
            if self.full_parameter_support:
                utility_parameter_samples = self.utility.parameter_distribution.support
            else:
                utility_parameter_samples = self.utility.parameter_distribution.sample(
                    self.number_of_utility_parameter_samples)
            for i in range(len(utility_parameter_samples)):
                marginal_argmax = self._current_marginal_argmax(utility_parameter_samples[i])
                X_baseline.append(marginal_argmax[0, :])
            X_baseline = np.atleast_2d(X_baseline)
            fX_baseline = f(X_baseline)[:, 0]
            anchor_points = np.vstack((anchor_points, X_baseline))
            anchor_points_values = np.concatenate((anchor_points_values, fX_baseline))
        print('Anchor points:')
        print(anchor_points)
        print('Anchor points values:')
        print(anchor_points_values)

        if self.parallel:
            pool = Pool(8)
            optimized_points = pool.map(self._parallel_optimization_wrapper, anchor_points)
            print('optimized points (parallel):')
            print(optimized_points)
        else:
            optimized_points = [apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]                 
            print('Optimized points (sequential):')
            print(optimized_points)                
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        print('Acquisition value of selected point: {}'.format(-np.squeeze(fx_min)))
        return x_min, fx_min
    
    def _current_marginal_argmax(self, parameter):
        """
        Computes argmax E_n[U(f(x))|U] (The abuse of notation can be misleading; note that the expectation is with
        respect to the posterior distribution on f after n evaluations)
        """
        if self.utility.affine:
            if self.model.output_dim == 1:
                def val_func(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        valX += np.reshape(parameter*muX, (X.shape[0], 1))
                    return -valX

                def val_func_with_gradient(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    dval_dX = np.zeros(X.shape)
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        dmu_dX = self.model.posterior_mean_gradient(X)
                        valX += np.reshape(parameter*muX, (X.shape[0], 1))
                        dval_dX += np.reshape(parameter*dmu_dX, X.shape)
                    return -valX, -dval_dX
            else:              
                def val_func(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        valX += np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                    return -valX
    
                def val_func_with_gradient(X):
                    X = np.atleast_2d(X)
                    valX = np.zeros((X.shape[0], 1))
                    dval_dX = np.zeros(X.shape)
                    for h in range(self.number_of_gp_hyps_samples):
                        self.model.set_hyperparameters(h)
                        muX = self.model.posterior_mean(X)
                        dmu_dX = self.model.posterior_mean_gradient(X)
                        valX += np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                        dval_dX += np.tensordot(parameter, dmu_dX, axes=1)
                    return -valX, -dval_dX

        elif self.expectation_utility is not None:
            def val_func(X):
                X = np.atleast_2d(X)
                func_val = np.empty((X.shape[0], 1))
                for h in range(self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    for i in range(X.shape[0]):
                        func_val[i,0] += self.expectation_utility.func(parameter, mean[:, i], var[:, i])
                return -func_val

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                func_gradient = np.zeros(X.shape)
                for h in range(self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    dmean_dX = self.model.posterior_mean_gradient(X)
                    dvar_dX = self.model.posterior_variance_gradient(X)
                    aux = np.concatenate((dmean_dX,dvar_dX))
                    for i in range(X.shape[0]):
                        func_val[i,0] += self.expectation_utility.func(parameter, mean[:,i], var[:,i])
                        func_gradient[i,:] += np.matmul(self.expectation_utility.gradient(parameter,mean[:,i],var[:,i]),aux[:,i])
                return -func_val, -func_gradient

        argmax = self.optimize_inner_func(f=val_func, f_df=val_func_with_gradient)[0]
        return argmax

    def optimize_inner_func(self, f=None, df=None, f_df=None, duplicate_manager=None, n_starting=200, n_anchor=16):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df

        # Update the optimizer, in case context has beee passed.
        self.inner_optimizer = choose_optimizer(self.inner_optimizer_name, self.context_manager.noncontext_bounds)

        # Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, latin_design_type, f, n_starting)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        # Select the anchor points (with context)
        anchor_points, anchor_points_values = anchor_points_generator.get(num_anchor=n_anchor, duplicate_manager=duplicate_manager, context_manager=self.context_manager, get_scores=True)
        
        # Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)
        optimized_points = [apply_optimizer_inner(self.inner_optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        #x_min = np.atleast_2d(anchor_points[0])
        #fx_min = np.atleast_2d(anchor_points_values[0])       
        return x_min, fx_min

    def _parallel_optimization_wrapper(self, x0):
        return apply_optimizer(self.optimizer, x0, self.f, None, self.f_df)


class ContextManager(object):
    """
    class to handle the context variable in the optimizer
    :param space: design space class from GPyOpt.
    :param context: dictionary of variables and their contex values
    """

    def __init__(self, space, context = None):
        self.space = space
        self.all_index = list(range(space.model_dimensionality))
        self.all_index_obj = list(range(len(self.space.config_space_expanded)))
        self.context_index = []
        self.context_value = []
        self.context_index_obj = []
        self.nocontext_index_obj = self.all_index_obj
        self.noncontext_bounds = self.space.get_bounds()[:]
        self.noncontext_index = self.all_index[:]

        if context is not None:
            #print('context')

            ## -- Update new context
            for context_variable in context.keys():
                variable = self.space.find_variable(context_variable)
                self.context_index += variable.index_in_model
                self.context_index_obj += variable.index_in_objective
                self.context_value += variable.objective_to_model(context[context_variable])

            ## --- Get bounds and index for non context
            self.noncontext_index = [idx for idx in self.all_index if idx not in self.context_index]
            self.noncontext_bounds = [self.noncontext_bounds[idx] for idx in  self.noncontext_index]

            ## update non context index in objective
            self.nocontext_index_obj = [idx for idx in self.all_index_obj if idx not in self.context_index_obj]

    def _expand_vector(self,x):
        """"
        Takes a value x in the subspace of not fixed dimensions and expands it with the values of the fixed ones.
        :param x: input vector to be expanded by adding the context values
        """
        x = np.atleast_2d(x)
        x_expanded = np.zeros((x.shape[0],self.space.model_dimensionality))
        x_expanded[:,np.array(self.noncontext_index).astype(int)]  = x
        x_expanded[:,np.array(self.context_index).astype(int)]  = self.context_value
        return x_expanded
