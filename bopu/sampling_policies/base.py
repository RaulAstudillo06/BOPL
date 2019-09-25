# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)


class SamplingPolicyBase(object):
    """
    Base class for sampling policies in Bayesian optimization.

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain.
    """
    analytical_gradient_prediction = False

    def __init__(self, model, space):
        self.model = model
        self.space = space
        self.analytical_gradient_objective = self.analytical_gradient_prediction and self.model.analytical_gradient_prediction # flag from the model to test if gradients are available

    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        raise NotImplementedError('')