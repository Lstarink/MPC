import numpy as np
import math as mt


class StateSpace:
    def __init__(self, x0):
        self.x = x0

    def h(self, u):
        x_plus1 = 0
        self.state = x_plus1

    def g(self, u):
        y = 0
        return y

    def reset(self, x0):
        self.x = x0
