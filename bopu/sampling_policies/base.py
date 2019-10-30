# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)


class SamplingPolicyBase(object):
    """
    Base class for sampling policies in Bayesian optimization.

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain.
    """

    def __init__(self, model=None, space=None):
        self.model = model
        self.space = space

    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        raise NotImplementedError('')
