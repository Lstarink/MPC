
class PID_Controller:
    def __init__(self, kp, ki, kd, dt, saturate=False, upper_limit=None, lower_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.saturate = saturate
        self.error_d = 0
        self.error_i = 0
        self.prev_error = 0

    def Tick(self, error):
        self.UpdateErrors(error)
        tau = self.kp*error + self.ki*self.error_i + self.kd*self.error_d
        if self.saturate:
            return self.Saturate(tau)
        return tau

    def UpdateErrors(self, error):
        self.error_d = (error-self.prev_error)/self.dt
        self.error_i += error
        self.previous_error = error

    def Saturate(self, tau):
        if self.upper_limit:
            if tau > self.upper_limit:
                tau = self.upper_limit
        if self.lower_limit:
            if tau < self.lower_limit:
                tau = self.lower_limit
        return tau

