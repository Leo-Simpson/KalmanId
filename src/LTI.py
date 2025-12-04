import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are
import numpy.linalg as LA

class LTI_Class:
    def __init__(self, f_fn, g_fn):
        self.A, self.Bu, self.Bw = linearize(f_fn)
        self.C, self.Du, self.Dw = linearize(g_fn)

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nu = self.Bu.shape[1]
        self.nw = self.Bw.shape[1]

    def simulate(self, ws, us=None):
        if us is None:
            us = np.zeros((len(ws), self.nu))
        assert len(ws) == len(us), f"Dimension of us: {us.shape}, dimension of ws: {ws.shape}"
        N = len(us)
        xk = np.zeros(self.nx)
        ys = np.zeros((N, self.ny))
        for k in range(N):
            ys[k] = self.C @ xk + self.Du @ us[k] + self.Dw @ ws[k]
            xk = self.A @ xk + self.Bu @ us[k] + self.Bw @ ws[k]
        return ys

    def E(self, L, ys, us=None):
        N = len(ys)
        if us is None:
            us = np.zeros((N, self.nu))
        xk = ca.SX.zeros(self.nx)
        es = ca.SX.zeros((self.ny, N))
        for k in range(N):
            yhat = self.C @ xk + self.Du @ us[k]
            next_xk = self.A @ xk + self.Bu @ us[k]
            es[:, k] = ys[k] - yhat
            xk = next_xk + L @ es[:, k]
        return es.T

    # Function to compute the true Kalman gain
    def KalmanGain(self, W=None):
        """
            Compute the Kalman gain for the system
        """
        if W is None:
            W = np.eye(self.nw)
        Q = self.Bw @ W @ self.Bw.T  # E(w w^T)
        R = self.Dw @ W @ self.Dw.T  # E(v v^T)
        cross_covar = self.Bw @ W @ self.Dw.T  # E(w v^T)

        # Now compute the Kalman gain using the GDARE
        X = solve_discrete_are(self.A.T, self.C.T, Q, R, s=cross_covar)
        S = self.C @ X @ self.C.T + R
        L = (self.A @ X @ self.C.T + cross_covar) @ LA.inv(S)
        return L, S

    # Function to compute some Kalman gain when the true covariances are unknown (for initial guesses).
    def Kalman_gain_unknown_noise(self, Q, R):
        """
            Compute the Kalman gain for the system when noise covariances are known
        """
        # Now compute the Kalman gain using the GARE
        X = solve_discrete_are(self.A.T, self.C.T, Q, R)
        S = self.C @ X @ self.C.T + R
        L = self.A @ X @ self.C.T @ LA.inv(S)
        return L


from scipy.linalg import solve_discrete_lyapunov    
# the following function is only used for selecting feasible initial guesses, and for plotting the feasible set
def constraint(A, alpha, rho_max=0.999):
    """
        Stability constraint function constraint(A) = 1 - alpha * Trace(P-I) where P is the solution to the discrete Lyapunov equation
            P = A P A^T + I
        Doing so, the constraint reads constraint(A) > 0

        There is also a prior check to verify that the system is stable, i.e. rho(A) < rho_max
    """
    if max(abs(np.linalg.eigvals(A))) > rho_max:
        return -1.
    else:
        I = np.eye(A.shape[0])
        P = solve_discrete_lyapunov(A, I)
        return 1 - alpha*np.trace(P-I)


def linearize(fn):
    J = fn.jacobian()
    x0s = [
        ca.DM.zeros(fn.size_in(i)[0])
        for i in range(3)
    ]
    J_eval = J(x0s[0], x0s[1], x0s[2], 0)
    return J_eval[0].full(), J_eval[1].full(), J_eval[2].full()
