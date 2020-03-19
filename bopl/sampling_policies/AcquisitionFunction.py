# Copyright (c) 2019, Raul Astudillo

from sampling_policies.base import SamplingPolicyBase
from optimization_services.acquisition_optimizer import ContextManager


class AcquisitionFunction(SamplingPolicyBase):
    """
    This sampling policy chooses the next point according to an acquisition function.

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    """
    analytical_gradient_prediction = True

    def __init__(self, model, space, acquisition_func, evaluator):
        super(AcquisitionFunction, self).__init__(model, space)
        self.acquisition_func = acquisition_func
        self.evaluator = evaluator

    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        self.acquisition_func.update_samples()
        #try:
            #self.acquisition_func.update_samples()
        #except:
            #pass
        # Update the context if any
        self.acquisition_func.optimizer.context_manager = ContextManager(self.space, None)
        # We zip the value in case there are categorical variables
        return self.space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=None))
