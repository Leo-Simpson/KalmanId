import casadi as ca
import numpy as np
from LTI import LTI_Class

def model_1state(a, L, S=1):
    """
    The system is
        x+ = a * x + L * e
        y  = x + e
    where e ~ N(0,S)

    The variable w is such that it has unit covariance
    """
    x = ca.SX.sym("x")
    u = ca.SX.sym("u")
    w = ca.SX.sym("w")
    e = np.sqrt(S) * w
    xplus = a * x + L * e
    y = x + e
    f_fn = ca.Function("f", [x, u, w], [xplus])
    g_fn = ca.Function("g", [x, u, w], [y])
    return LTI_Class(f_fn, g_fn)

def model_2states(h, mu, sigma_f):
    """
        This model is for the physical system:
        ddot{q}(t) = - mu * dot{q}(t) + f(t)
        y_k = q(t_k) + v_k ,  v_k ~ N(0, 1)
        and f(t) is piecewise constant:
        f(t) = sigma_f * w_k ,  t in [t_k, t_{k+1}), with t_{k+1} - t_k = dt,
        with constant intervals h, and w_k ~ N(0, 1)

        This model can integrated in an exact way:
            dot{q}(t_k+1) = a * dot{q}(t_k) + b * sigma_f * w_k
            q(t_k+1) = q(t_k) + b * dot{q}(t_k) + c * sigma_f * w_k
        with
            a = exp(- mu * h)
            b = (1 - exp(- mu * h)) / mu
            c = (h - b) / mu

    """
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u") # not used
    w = ca.SX.sym("w")
    v = ca.SX.sym("v")
    noise = ca.vertcat(w, v)

    q, qdot = x[0], x[1]

    a = np.exp(- mu * h)
    b = (1 - a) / mu
    c = (h - b) / mu

    qdot_plus = a * qdot + b * sigma_f * w
    q_plus = q + b * qdot + c * sigma_f * w
    xplus = ca.vertcat(q_plus, qdot_plus)
    y = q + v

    f_fn = ca.Function("f", [x, u, noise], [xplus])
    g_fn = ca.Function("g", [x, u, noise], [y])

    return LTI_Class(f_fn, g_fn)

def model_3states(h, mu, r1, r2, af, p):
    """
        This model is for the physical system:
            ddot{q}(t) = - mu * dot{q}(t) + f(t)
        and f(t) is piecewise constant:
            f(t) = f_k ,  t in [t_k, t_{k+1}), with constant intervals h,
        But now, f_k follows its own dynamics:
            f_{k+1} = af * f_k + w_k ,
        where w_k is a random variable, which is modelled as a mixture, with rare events (outside of this function)
        The observations are:
            y_k = [ q(t_k); ddot{q}(t_k) ] + v_k ,  v_k ~ N(0, diag([r1, r2]))

        Again, this model can integrated in an exact way:
            dot{q}(t_k+1) = a * dot{q}(t_k) + b * f_k
            q(t_k+1) = q(t_k) + b * dot{q}(t_k) + c * f_k
    """
    x = ca.SX.sym("x", 3)
    u = ca.SX.sym("u") # not used
    w = ca.SX.sym("w")
    v = ca.SX.sym("v", 2)
    noise = ca.vertcat(w, v)

    q, qdot, f = x[0], x[1], x[2]

    a = np.exp(- mu * h)
    b = (1 - a) / mu
    c = (h - b) / mu

    qdot_plus = a * qdot + b *  f
    q_plus = q + b * qdot + c * f
    fplus = af * f + w
    xplus = ca.vertcat(q_plus, qdot_plus, fplus)
    y = ca.vertcat(q, f - mu*qdot) + np.sqrt([r1, r2]) * v

    f_fn = ca.Function("f", [x, u, noise], [xplus])
    g_fn = ca.Function("g", [x, u, noise], [y])


    # Create a way to generate noise
    def generate_noise(rng, N):
        # first generate a Bernoulli random variable with parameter p
        bernoulli_samples = rng.uniform(0, 1, N) < p
        w_process = bernoulli_samples * rng.normal(0, 1, size=N) / np.sqrt(p)
        w_measurement = rng.normal(0, 1., size=(N, 2))
        w = np.concatenate((w_process[:, None], w_measurement), axis=1)
        return w
    W = np.eye(3) # by construction, w has variance I_3

    return LTI_Class(f_fn, g_fn), generate_noise, W
