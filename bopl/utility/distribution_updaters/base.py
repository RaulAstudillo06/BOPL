# Copyright (c) 2019, Raul Astudillo Marban


class DistributionUpdaterBase(object):
    """
    Base class to handle the updater of the utility's distribution.
    """

    def __init__(self, preference_elicitation_strategy):
        self.preference_elicitation_strategy = preference_elicitation_strategy
    
    def eval_func_and_gradient(self, y, theta):
        """
        Samples random parameter from parameter distribution and evaluates the utility function and its gradient at y given this parameter.
        """
        utility_value = self.eval_func(y, theta)
        utility_gradient = self._eval_gradient(y, theta)
        return utility_value, utility_gradient
    
    def eval_func(self, y, theta):
        """
        Evaluates the utility function at y given a fixed parameter.
        """
        return self.func(y, theta)
    
    def eval_gradient(self, y, theta):
        """
        Evaluates the gradient f the utility function at y given a fixed parameter.
        """
        return self.gradient(y, theta)

    def sample_parameter(self, number_of_samples=1):
        return self.parameter_distribution.sample(number_of_samples)

    def update_parameter_distribution(self, Y):
        if self.distribution_updater_available:
            self.parameter_distribution.update_distribution(Y)
        else:
            print('Utility distribution updater has not been provided')
