import casadi as ca
import numpy as np

# Now this is for the solving the optimization problem
# Options for the optimization solver (this could also be set as a parameter of the functions if needed)
default_opts = {
    "L2_pen": 1.0, # Penalty on the step size
    "max_iters": 1500,
    "tol": 1e-4,
    "tau": 1e-3,
    "maxiter_line_search": 100,
    "beta_line_search": 0.8,
    "c_armijo": 0.3,
}

def InteriorPointSolver(scale, symbolics, x0, opts={}):
    """
        Solve the optimization problem
            min_x scale * 0.5 * ||r(x)||^2
            s.t. h(x) >= 0
                 P(x) is positive definite
    
        using (primal-dual/log-barrier) interior point method with line-searched globalization approach and GN Hessian approximation.

        Note that the positive definiteness of P(x) is only checked during line-search, not enforced as a constraint, because we assume that h(x) already acts as a barrier for it.
    """
    opts = {**default_opts, **opts} # Merge default options with user options
    J_sym = ca.jacobian( symbolics['r'], symbolics['x'])
    value = 0.5 * scale * ca.sumsqr( symbolics['r'])
    gradient = scale * J_sym.T @ symbolics['r']
    hessian = scale * J_sym.T @ J_sym  + opts["L2_pen"] * ca.DM.eye(x0.shape[0])
    sqrtP_sym = ca.trace(ca.chol(symbolics['P'])) # this is regular (i.e. not nan) iff P is positive definite

    grad_h_sym = ca.jacobian(symbolics['h'], symbolics['x']).T
    merit_sym = value - opts["tau"] * ca.log(symbolics['h'])

    f_fn = ca.Function("f", [symbolics['x']], [value])
    merit_fn = ca.Function("merit", [symbolics['x']], [merit_sym])
    cheap_h = ca.Function("h", [symbolics['x']], [symbolics['h'], sqrtP_sym])
    expensive_h = ca.Function("h_full", [symbolics['x']], [symbolics['h'], grad_h_sym])
    expensive_f = ca.Function("grad", [ symbolics['x']], [gradient, hessian])

    stats = {"f_fn": f_fn, "success": False}
    x_k = x0.copy()
    mu_k = 1.0 # initial dual variable for the inequality constraint
    iterates = [x_k]
    for k in range(1, opts["max_iters"]+1):
        tau_k = opts["tau"]  # Note that usually, to get better convergence, tau should start bigger, and decrease along iterations. Here, we fix it to make the code simpler, but the convergence will be slower...
        hk, grad_hk = expensive_h(x_k)
        if hk < 0:
            stats["reason"] = f"iterate is not feasible"
            break
        grad_fk, H_fk = expensive_f(x_k) # This is in vector form
    
        # Now compute the step
        grad_tot_k = grad_fk - (tau_k / hk) * grad_hk # gradient of the merit function
        H_tot = H_fk + (mu_k / hk) * (grad_hk @ grad_hk.T)  # Hessian of the merit function
        dx_k = - ca.solve(H_tot, grad_tot_k) # Compute the step
        dhk = grad_hk.T @ dx_k
        dmu_k = (tau_k - mu_k * dhk  )/hk  - mu_k
        dmerit_k = grad_tot_k.T @ dx_k # this derivative is always negative

        if ca.norm_inf(dx_k) < opts["tol"]:
            stats["success"] = True
            break
        # Line search to ensure feasibility and sufficient decrease
        assert dmerit_k < 0, "Search direction is not a descent direction for the merit function."
        merit_k = merit_fn(x_k)
        step_size = 1.0
        for i in range(opts["maxiter_line_search"]):
            x_ = x_k + step_size * dx_k
            mu_ = mu_k + step_size * dmu_k
            h_, sqrtP_ = cheap_h(x_)
            if sqrtP_.is_regular() and h_ >= 0 and mu_ >= 0:
                merit_ = merit_fn(x_)
                if merit_ < merit_k + opts["c_armijo"] * step_size * dmerit_k:
                    break
            step_size = opts["beta_line_search"] * step_size
        if i == opts["maxiter_line_search"] - 1:
            stats["reason"] =  f"Globalisation failed at iteration {k}"
            break
        # Update the iterate
        x_k = x_k + step_size * dx_k # or x_k = x_candidate
        mu_k = mu_k + step_size * dmu_k
        iterates.append(x_k)

    stats["n_iters"] = k
    if k == opts["max_iters"]:
        # If you are here, that means that you did not converge
        stats["reason"] = "Maximum number of iterations reached"
    iterates = ca.hcat(iterates).T
    return iterates, stats
