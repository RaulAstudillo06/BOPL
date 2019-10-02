# Copyright (c) 2019, Raul Astudillo

import numpy as np
from sampling_policies.base import SamplingPolicyBase
from optimization_services.u_acquisition_optimizer import ContextManager
from aux_software.GPyOpt.experiment_design import initial_design
from aux_software.GPyOpt.optimization.optimizer import apply_optimizer, choose_optimizer
from copy import deepcopy


class uTS(SamplingPolicyBase):
    """
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer, utility):
        super(uTS, self).__init__(model, space)
        self.optimizer_name = optimizer
        self.utility = utility
        #
        self.context_manager = ContextManager(self.space)
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)
        self.X_aux = None
        self.Y_aux = None

    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        utility_parameter_sample = self.utility.sample_parameter(number_of_samples=1)
        model_sample = self.model.get_copy_of_model_sample()
        X_evaluated = np.copy(model_sample[0].X)
        Y_evaluated = []
        for j in range(len(model_sample)):
            Y_evaluated.append(np.copy(model_sample[j].Y))
        self.X_aux = np.copy(X_evaluated)
        self.Y_aux = deepcopy(Y_evaluated)

        def objective_func_sample(x):
            x_new = np.atleast_2d(x)
            y_new = []
            self.X_aux = np.vstack((self.X_aux, x_new))
            for j in range(len(model_sample)):
                y_new.append(model_sample[j].posterior_samples_f(x_new, size=1, full_cov=True)[0])
                self.Y_aux[j] = np.vstack((self.Y_aux[j], y_new[j]))
                model_sample[j].set_XY(self.X_aux, self.Y_aux[j])
            y_new = np.squeeze(np.asarray(y_new))
            val = self.utility.eval_func(y_new, utility_parameter_sample)
            return -val

        d0 = initial_design('random', self.space, 1)
        try:
            suggested_sample = apply_optimizer(self.optimizer, d0, f=objective_func_sample,
                                     context_manager=self.context_manager, space=self.space, maxfevals=1000)[0]
        except:
            suggested_sample = d0

        suggested_sample = np.atleast_2d(suggested_sample)
        return suggested_sample
