import numpy as np
import nonlinear_state_space_model

class FeedForwardController:
    def __init__(self, statespace):
        self.statespace = statespace

    def Tick(self, reference):
        x_lin = np.array([reference[0], reference[1], reference[2],
                          0, 0, 0])

        return 0
