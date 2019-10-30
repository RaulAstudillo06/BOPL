# Copyright (c) 2018, Raul Astudillo

import numpy as np
from copy import deepcopy

class BasicModel(object):
    """
    Class for handling a very simple model that only requires saving the evaluated points (along with their corresponding outputs) so far.
    """
    analytical_gradient_prediction = True

    def __init__(self, output_dim=None, X=None, Y=None):
        self.output_dim = output_dim
        self.X = X
        self.Y = Y
        self.name = 'basic model'

    def updateModel(self, X, Y):
        """
        Updates the model with new observations.
        """
        self.X = X
        self.Y = Y

    def get_X(self):
        return np.copy(self.X)

    def get_Y(self):
        return deepcopy(self.Y)

    def get_XY(self):
        X = np.copy(self.X)
        Y = deepcopy(self.Y)
        return X, Y
