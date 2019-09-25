# Copyright (c) 2019, Raul Astudillo

import numpy as np

class UtilityDistribution(object):
    """
    Class to handle the parameter distribution of the utility function.
    There are two possible ways to specify a parameter distribution: ...
    """
    def __init__(self, support=None, prob_dist=None, sample_generator=None, use_full_support=None, distribution_updater=None):
        if support is None and sample_generator is None:
            raise Exception('Either a finite support or a sample generator have to be provided.')
        self.support = support
        self.prob_dist = prob_dist
        self.sample_generator = sample_generator
        self.distribution_updater = distribution_updater
        if support is not None:
            self.support_cardinality = len(support)
        if use_full_support is not None:
            self.use_full_support = use_full_support
        else:
            if support is not None and self.support_cardinality < 20:
                self.use_full_support = True
            else:
                self.use_full_support = False
    
    def sample(self, number_of_samples=1):
        if self.support is None:
            parameter_samples = self.sample_generator(number_of_samples)
        else:
            indices = np.random.choice(int(len(self.support)), size=number_of_samples, p=self.prob_dist)
            parameter_samples = self.support[indices,:]
        return parameter_samples

    def update_parameter_distribution(self, Y):
        if self.distribution_updater is not None:
            updated_distribution = self.distribution_updater(self.support, self.prob_dist, self.sample_generator, Y)
            self.support = updated_distribution['support']
            self.prob_dist = updated_distribution['prob_dist']
            self.sample_generator = updated_distribution['sample_generator']
            print('Utility distribution has been updated')
        else:
            print('Utility distribution updater has not been provided')
