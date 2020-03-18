# Copyright (c) 2019, Raul Astudillo

from sampling_policies.base import SamplingPolicyBase
from aux_software.GPyOpt.experiment_design import initial_design


class Random(SamplingPolicyBase):
    """
    This sampling policy chooses the next point to sample uniformly at random over the optimization space.

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    """
    analytical_gradient_prediction = True

    def __init__(self, model, space):
        super(Random, self).__init__(model, space)
            
    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        suggested_sample = initial_design('random', self.space, 1)
        return suggested_sample
