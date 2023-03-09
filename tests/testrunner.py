import unittest

import test_kinematics
import test_dynamics

def RunAllTests():
    verb = 2

    kinematics = unittest.TestLoader().loadTestsFromModule(test_kinematics)
    unittest.TextTestRunner(verbosity=verb).run(kinematics)

    dynamics = unittest.TestLoader().loadTestsFromModule(test_dynamics)
    unittest.TextTestRunner(verbosity=verb).run(dynamics)

if __name__ == "__main__":
    RunAllTests()