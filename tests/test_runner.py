import unittest
import test_nonlinear_state_space_model
import test_LQR


def RunAllTests():
    verb = 2

    state_space = unittest.TestLoader().loadTestsFromModule(test_nonlinear_state_space_model)
    unittest.TextTestRunner(verbosity=verb).run(state_space)

    lqr = unittest.TestLoader().loadTestsFromModule(test_LQR)
    unittest.TextTestRunner(verbosity=verb).run(lqr)


if __name__ == "__main__":
    RunAllTests()
