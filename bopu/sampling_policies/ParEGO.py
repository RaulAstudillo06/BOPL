
#import os
#script_dir = os.path.dirname(os.path.abspath(__file__))
#import sys
#sys.path.append(script_dir[:-18])
#print(script_dir[:-18])
from sampling_policies.base import SamplingPolicyBase
from core import MultiOutputGP
import aux_software.GPyOpt as GPyOpt
from utility import UtilityDistribution
from utility import Utility
from sampling_policies.AcquisitionFunction import AcquisitionFunction
from optimization_services import U_AcquisitionOptimizer
from sampling_policies.acquisition_functions import uEI_affine
from core import chebyshev_scalarization
from core import preference_encoder
from bopu import BOPU
import numpy as np
from copy import deepcopy


class ParEGO(SamplingPolicyBase):
    """
    This sampling policy chooses the next point to sample uniformly at random over the optimization space.

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    """
    analytical_gradient_prediction = True

    def __init__(self, model, space, utility):
        super(ParEGO, self).__init__(model, space)
        self.utility = utility
        self.preference_information = self.utility.parameter_distribution.preference_information

    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        # Get current evaluations
        model_sample = self.model.get_copy_of_model_sample()
        X_evaluated = np.copy(model_sample[0].X)
        Y_evaluated = []
        for j in range(len(model_sample)):
            Y_evaluated.append(np.copy(model_sample[j].Y))
        self.X_aux = np.copy(X_evaluated)
        self.Y_aux = deepcopy(Y_evaluated)

        # Auxiliary Bayesian optimization model to run ParEGO
        weight = np.random.dirichlet(np.ones(len(self.Y_aux)))#self._sample_posterior_weight_for_chebyshev_scalarization()
        self.Y_aux = np.reshape(self.Y_aux, (len(model_sample), self.Y_aux[0].shape[0]))
        scalarized_fX = chebyshev_scalarization(self.Y_aux, weight)
        scalarized_fX = np.reshape(scalarized_fX, (len(scalarized_fX), 1))

        aux_model = MultiOutputGP(output_dim=1, exact_feval=[True], fixed_hyps=False)

        def aux_utility_func(parameter, y):
            return np.dot(parameter, y)

        def aux_utility_gradient(parameter, y):
            return parameter

        aux_utility_parameter_support = np.ones((1, 1))
        aux_utility_parameter_prob_distribution = np.ones((1, ))
        aux_utility_param_distribution = UtilityDistribution(support=aux_utility_parameter_support, prob_dist=aux_utility_parameter_prob_distribution, utility_func=aux_utility_func)
        aux_utility = Utility(func=aux_utility_func, gradient=aux_utility_gradient, parameter_distribution=aux_utility_param_distribution, affine=True)

        aux_acquisition_optimizer = U_AcquisitionOptimizer(space=self.space, model=aux_model, utility=aux_utility, optimizer='lbfgs')

        aux_acquisition = uEI_affine(aux_model, self.space, optimizer=aux_acquisition_optimizer, utility=aux_utility)
        aux_evaluator = GPyOpt.core.evaluators.Sequential(aux_acquisition)
        aux_sampling_policy = AcquisitionFunction(aux_model, self.space, aux_acquisition, aux_evaluator)
        bopu = BOPU(aux_model, self.space, sampling_policy=aux_sampling_policy, utility=aux_utility, X_init=self.X_aux, Y_init=[scalarized_fX], dynamic_utility_parameter_distribution=False)
        suggested_sample = bopu.suggest_next_point_to_evaluate()
        return suggested_sample

    def _sample_posterior_weight_for_chebyshev_scalarization(self):
        if len(self.preference_information) == 0:
            weight = np.random.dirichlet(np.ones(len(self.Y_aux)))
        else:
            test_counter = 0
            posterior_wight_found = False
            while not posterior_wight_found:
                suggested_weight = np.random.dirichlet(np.ones(len(self.Y_aux)))
                test_counter += 1
                keep_verifying = True
                counter = 0
                while keep_verifying and counter < len(self.preference_information):
                    preference_information_item = self.preference_information[counter]
                    u1 = chebyshev_scalarization(preference_information_item[0], suggested_weight)
                    u2 = chebyshev_scalarization(preference_information_item[1], suggested_weight)
                    pref = preference_encoder(u1, u2)
                    if not pref == preference_information_item[2]:
                        keep_verifying = False
                    counter += 1
                if counter == len(self.preference_information):
                    weight = suggested_weight
                    posterior_wight_found = True
            weight = np.atleast_1d(weight)
            print('Required parameter samples = {}'.format(test_counter))
        return weight
