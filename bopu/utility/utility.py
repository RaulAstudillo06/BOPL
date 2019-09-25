# Copyright (c) 2019, Raul Astudillo Marban


class Utility(object):
    """
    Class to handle the utility function's statistical model (along with gradients if available).

    param func: utility function.
    param dfunc: gradient of the utility function (if available).
    param parameter_space: space of parameters (Theta) of the utility function.
    param parameter_dist: distribution over the spaceof parameters.
    param linear: whether utility function is linear or not (this is used to save computations later; default, False)

    .. Note:: .
    """

    def __init__(self, func, gradient=None, parameter_distribution=None, expectation=None, affine=False):
        self.func = func
        self.gradient = gradient
        self.parameter_distribution = parameter_distribution
        self.expectation = expectation
        self.affine = affine
        if parameter_distribution.distribution_updater is not None:
            self.distribution_updater_available = True
        else:
            self.distribution_updater_available = False
    
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
