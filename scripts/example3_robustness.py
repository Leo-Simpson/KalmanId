import numpy as np
import sys
from os.path import join, dirname
main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)

from Models import model_3states
from PEM import solve_PEM
from LTI import constraint


h, mu, r1, r2, af, p = 0.1, 0.5, 10., 1., 0.9, 0.1
model, generate_w, W = model_3states(h, mu, r1, r2, af, p)

Nlist = [100]
alpha = 0.02
rng = np.random.default_rng(42)
ws = generate_w(rng, Nlist[-1])
ys = model.simulate(ws)

nL0 = 60
sqrtQ =  rng.uniform(-1, 1, size=(nL0, model.nx, model.nx))
sqrtR =  rng.uniform(-1, 1, size=(nL0, model.ny, model.ny))
L0s= []
for i in range(nL0):
    Q = sqrtQ[i] @ sqrtQ[i].T
    R = sqrtR[i] @ sqrtR[i].T
    L0 = model.Kalman_gain_unknown_noise(Q, R)
    if constraint(model.A - L0 @ model.C, alpha) >= 0:
        L0s.append(L0)
L0s = np.array(L0s)
print(f"Number of stable initial guesses: {L0s.shape[0]} out of {nL0}")
initial_guesses = np.array( [L0.flatten(order='F') for L0 in L0s] )

dict_solutions = {}
for N in Nlist:
    print(f"\n \n -------- Computing solutions for N={N}")
    L_hats = []
    for i, L0 in enumerate(L0s):
        print(f"Solving for initial guess {i+1}/{L0s.shape[0]}")
        L_hat, stats = solve_PEM(model, ys[:N], L0, alpha)
        assert stats['success'], f"Solver failed with message: {stats['return_status']}"
        L_hats.append(L_hat)
    solutions = np.array( [L_hat.flatten(order='F') for L_hat in L_hats] )
    dict_solutions[N] = solutions

# Now display the important information
print("\n \n -------- Summary of results:")
print(f"Number of initial guesses: {initial_guesses.shape[0]}")
def compute_pairwise_distances(vectors):
    # the array should have shape (n_samples, n_features)
    diffs = vectors[:, None, :] - vectors[None, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=2))
    return distances[np.triu_indices(distances.shape[0], k=1)]
initial_distances = compute_pairwise_distances(initial_guesses)

print(f" Average distance between initial guesses: {np.mean(initial_distances):.2e}")
for i, (N, final) in enumerate(dict_solutions.items()):
    assert len(initial_guesses) == len(final), "Initial and final guesses must have the same length"
    final_distances = compute_pairwise_distances(final)
    print(f"\n For N={N}:")
    print(f"  Maximum distance between the solutions: {np.mean(final_distances):.2e}")
