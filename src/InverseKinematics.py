import numpy as np
import pid_controller
import config

class InverseKinematics:
    def __init__(self, kp, ki, kd, dt, n_controllers=8, saturate=False, upper_limit=None, lower_limit=None):
        self.controllers = np.zeros([n_controllers],dtype=object)
        self.n_controllers = n_controllers
        for i in range(n_controllers):
            self.controllers[i] = pid_controller.PID_Controller(kp, ki, kd, dt,
                                                                saturate=saturate,
                                                                upper_limit=upper_limit,
                                                                lower_limit=lower_limit)

    def Tick(self, reference, location):
        desired_lengths = InverseKinematics.InverseKinematics(self, reference)
        actual_lengths = InverseKinematics.InverseKinematics(self, location)
        error_vector = actual_lengths-desired_lengths
        print("Error Vector: ", error_vector)
        tau_vector = np.zeros(self.n_controllers)
        for n, error in enumerate(error_vector):
            tau_vector[n] = self.controllers[n].Tick(error)

        return tau_vector, error_vector


    def InverseKinematics(self, location):
        l = np.zeros([self.n_controllers])
        attachement_points = config.attachment_points

        for n, point in enumerate(attachement_points):
            l[n] = np.linalg.norm((location-point))
        return l


