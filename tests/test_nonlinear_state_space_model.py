import unittest
import numpy as np
import math as mt
import nonlinear_state_space_model
import matplotlib.pyplot as plt
import visualize


class TestStateSpace(unittest.TestCase):
    def test_nonlinear_state_space_model_EvaluateNonlinear(self):
        state_space = nonlinear_state_space_model.StateSpaceModel()
        f_evaluated1 = state_space.EvaluateNonlinear(np.zeros(6),
                                                     np.zeros(8))
        f_evaluated2 = state_space.EvaluateNonlinear(np.zeros(6),
                                                     np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        f_evaluated3 = state_space.EvaluateNonlinear(np.array([1, 0, 0, 0, 0, 0]),
                                                     np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        # zero input zero output
        self.assertEqual(np.linalg.norm(f_evaluated1), 9.81)
        # pull 4 strings located at the positive x side to generate acceleration in x direction and nothing else
        self.assertEqual(f_evaluated2[3], 4/mt.sqrt(3))
        # pull 4 strings located at the positive x side, with the mass in the same plane, so no acceleration
        self.assertEqual(np.linalg.norm(f_evaluated3 - f_evaluated1), 0)

    def test_nonlinear_state_space_model_Linearize(self):
        statespace = nonlinear_state_space_model.StateSpaceModel()
        statespace.Linearize()
        #linearized around 0
        correctA = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        correctB1_lower_absolute = np.ones([3,8])/np.sqrt(3)

        self.assertIsNone(np.testing.assert_array_equal(correctA, statespace.A))
        self.assertIsNone(np.testing.assert_array_equal(np.zeros([3, 8]), statespace.B[0:3, :]))
        self.assertIsNone(np.testing.assert_array_equal(correctB1_lower_absolute, abs(statespace.B[3:6, :])))

        statespace.Linearize(x_eq=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        oversqrt2 = 1/np.sqrt(2)
        oversqrt6 = 1/np.sqrt(6)

        correctB2_lower_absolute = np.array([[0.0, 0.0, 0.0, 0.0, 2*oversqrt6, 2*oversqrt6, 2*oversqrt6, 2*oversqrt6],
                                             [oversqrt2, oversqrt2, oversqrt2, oversqrt2, oversqrt6, oversqrt6, oversqrt6, oversqrt6],
                                             [oversqrt2, oversqrt2, oversqrt2, oversqrt2, oversqrt6, oversqrt6, oversqrt6, oversqrt6]])

        self.assertIsNone(np.testing.assert_array_equal(correctA, statespace.A))
        self.assertIsNone(np.testing.assert_array_equal(np.zeros([3, 8]), statespace.B[0:3, :]))
        self.assertIsNone(np.testing.assert_array_equal(correctB2_lower_absolute, abs(statespace.B[3:6, :])))

        statespace.Linearize(x_eq=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), u_eq=np.random.rand(8))

        self.assertIsNone(np.testing.assert_array_equal(np.zeros([3, 8]), statespace.B[0:3, :]))
        self.assertIsNone(np.testing.assert_array_equal(correctB2_lower_absolute, abs(statespace.B[3:6, :])))

    def test_nonlinear_state_space_model_dt_state_space(self):
        statespace = nonlinear_state_space_model.StateSpaceModel()
        f_eq = 9.81/(4*mt.sqrt(3))
        u_eq = np.array([0, f_eq, 0, f_eq, 0, f_eq, 0, f_eq])
        statespace.Linearize(u_eq=u_eq)
        dt = 0.01
        t = np.arange(0, 10, dt)
        N = len(t)
        u = np.zeros([statespace.m, N])
        u[0] = np.sin(t/5)
        u[1] = np.cos(t/5)
        u[2] = -np.sin(t/5)
        u[3] = -np.cos(t/5)
        y1 = statespace.ct_sim(u.T, t, np.zeros(statespace.n))
        y1 = np.array(y1)
        y1 = y1.T
        # visualize.VisualizeTrajectory3D(y1[:][0], y1[:][1], y1[:][2])
        # [fig, axs] = visualize.VisualizeStatePrograssion(y1, t)


        statespace.Setdt(dt)
        y2 = statespace.dt_sim(u.T, t, np.zeros(statespace.n))
        y2 = np.array(y2)
        y2 = y2.T
        print(y2.shape)
        print(y2[:][0].shape)
        # visualize.VisualizeTrajectory3D(y2[:][0], y2[:][1], y2[:][2])
        # visualize.VisualizeStatePrograssion(y2, t)

        y3 = statespace.nl_sim(u.T, t, np.zeros(statespace.n))
        print(y3.shape)
        visualize.VisualizeStatePrograssion(y3, t)

        # floatingpointerror = 1e-1
        # self.assertIsNone(np.testing.assert_allclose(y1[:][0], y2[:][0], atol=floatingpointerror))
        # self.assertLess(np.linalg.norm(y1[:][1] - y2[:][1]), floatingpointerror)
        # self.assertIsNone(np.testing.assert_allclose(y1[:][1], y2[:][1], atol=floatingpointerror))
        # self.assertIsNone(np.testing.assert_allclose(y1[:, 2], y2[:, 2], atol=floatingpointerror))
        # self.assertIsNone(np.testing.assert_allclose(y1[:, 3], y2[:, 3], atol=floatingpointerror))
        # self.assertIsNone(np.testing.assert_allclose(y1[:, 4], y2[:, 4], atol=floatingpointerror))
        # self.assertIsNone(np.testing.assert_allclose(y1[:, 5], y2[:, 5], atol=floatingpointerror))

    # def test_ConvinceU(self):
    #     statespace = nonlinear_state_space_model.StateSpaceModel()
    #     statespace.Linearize(x_eq=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    #     A1 = statespace.A
    #     B1 = statespace.B
    #
    #     statespace.Linearize(x_eq=np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
    #                          u_eq=np.array([0.1, 0.1, 0.1, 0.1,
    #                                         0.0, 0.0, 0.00, 0.0]))
    #
    #     A2 = statespace.A
    #     B2 = statespace.B
    #
    #
    #
    #     fig, axs = plt.subplots(2)
    #     fig.suptitle('Vertically stacked subplots')
    #     cax1 = axs[0].matshow(B1, interpolation='nearest')
    #     fig.colorbar(cax1)
    #     cax2 = axs[1].matshow(B2, interpolation='nearest')
    #     fig.colorbar(cax2)
    #     axs[0].matshow(B1)
    #     axs[1].matshow(B2)
    #     fig.savefig('foo.png', bbox_inches='tight')
    #
    #     fig, axs = plt.subplots(2)
    #     fig.suptitle('Vertically stacked subplots')
    #     cax1 = axs[0].matshow(A1, interpolation='nearest')
    #     fig.colorbar(cax1)
    #     cax2 = axs[1].matshow(A2, interpolation='nearest')
    #     fig.colorbar(cax2)
    #     axs[0].matshow(A1)
    #     axs[1].matshow(A2)
    #     fig.savefig('foo2.png', bbox_inches='tight')
    #
    #
