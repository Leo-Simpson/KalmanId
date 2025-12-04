import casadi as ca
from Optimization import InteriorPointSolver

def create_Lyapunov_solver(n):
    """
        Create a casadi function that solves the discrete-time Lyapunov equation:
            P = A P A^T + I
        for a given square matrix A of size n x n.
    """
    A = ca.SX.sym("A", n, n)
    P = ca.SX.sym("P", n, n)
    I = ca.DM.eye(n)
    Operator =  P - A @ P @ A.T
    Matrix_operator = ca.jacobian(ca.vec(Operator), ca.vec(P))
    solution_vec = ca.solve(Matrix_operator, ca.vec(I))
    solution = ca.reshape(solution_vec, (n, n))
    solution_noP = ca.Function("lyap_fn", [A, P], [solution])(A, ca.DM.zeros((n, n)))
    lyap_fn = ca.Function("lyap_fn", [A], [solution_noP])
    return lyap_fn

def formulate_PEM(model, ys, alpha, us=None):
    # Symbolic functions for the residuals
    L_mat = ca.SX.sym("L", model.nx, model.ny)
    residuals = model.E(L_mat, ys, us=us).reshape((-1, 1))
    x = L_mat.reshape((-1, 1))
    Lyapunov_fn = create_Lyapunov_solver(model.nx)
    P_sym = Lyapunov_fn(model.A - L_mat @ model.C)
    h_sym = 1 - alpha * ca.trace(P_sym-ca.DM.eye(model.nx))
    return {'x': x, 'r': residuals, 'h': h_sym, 'P': P_sym}

def solve_PEM(model, ys, L0, alpha, us=None, raise_errors=True, opts={}):
    # Symbolic functions for the residuals
    x0 = L0.flatten(order='F')
    scale = 1.0 / len(ys)
    symbolics = formulate_PEM(model, ys, alpha, us=us)
    iterates, stats = InteriorPointSolver(scale, symbolics, x0, opts=opts)
    stats["iterates"] = iterates.full()
    L_sol = iterates[-1, :].reshape((model.nx, model.ny)).full()
    if stats["success"]:
        print(f"Success: Interior point method converged after {stats['n_iters']} iterations.")
    else:
        message = f"Failure: {stats['reason']} after {stats['n_iters']} iterations."
        if raise_errors:
            raise RuntimeError(message)
        else:
            print(message)
    return L_sol, stats

