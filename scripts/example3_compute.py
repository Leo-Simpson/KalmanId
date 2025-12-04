import numpy as np
import pickle
import sys
from os.path import join, dirname
main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)

from Models import model_3states
from PEM import solve_PEM

h, mu, r1, r2, af, p = 0.1, 0.5, 2., 1., 0.9, 0.1
model, generate_noise, W = model_3states(h, mu, r1, r2, af, p)

Nlist =  [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
L_star, S_star = model.KalmanGain(W=W)
x_star = L_star.flatten(order='F')

alpha = 0.02
n_realisations = 5
L0 = model.Kalman_gain_unknown_noise(np.eye(model.nx), np.eye(model.ny))
all_errors = []
rng = np.random.default_rng(42)
for i in range(n_realisations):
    print(f"\n \n \n -------- Solving for Realisation = {i+1}/{n_realisations} ------- \n \n \n")
    ws = generate_noise(rng, Nlist[-1])
    ys = model.simulate(ws)
    errors = []
    for N in Nlist:
        print(f"\n \n ---- Solving for N = {N} ---- ")
        L_hat, stats = solve_PEM(model, ys[:N], L0, alpha)
        assert stats['success'], f"Solver failed with message: {stats['return_status']}"
        x_hat = L_hat.flatten(order='F')
        error = abs(x_hat - x_star).max()
        print(f"  --> Error found {error:.2e}")
        errors.append(error)
    all_errors.append(errors)

dict_to_save = {
    "Nlist": Nlist,
    "all_errors": all_errors,
}
file_saving = join(main_dir, "savings", "consistency_data.pkl")
with open(file_saving, 'wb') as f:
    pickle.dump(dict_to_save, f)


