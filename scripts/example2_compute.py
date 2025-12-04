import numpy as np
import pickle
import sys
from os.path import dirname, join
main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)

from Models import model_2states
from PEM import solve_PEM
from LTI import constraint

h, mu, sigma_f = 0.1, 0.1, 10
model = model_2states(h, mu, sigma_f)

N_list = [10, 50, 100, 500]
alpha = 0.02
L1_range = [-0.1, 2.2]
L2_range = [-0.2, 3.]

rng = np.random.default_rng(42)
ws = rng.normal(0, 1, size=(N_list[-1], model.nw))
ys = model.simulate(ws)

ntrials = 100

L10 = rng.uniform(*L1_range, size=ntrials)
L20 = rng.uniform(*L2_range, size=ntrials)
L0s_all = np.array([L10, L20]).T[..., None]

# Filter only the stable initial gains
L0s = []
for i, L0 in enumerate(L0s_all):
    if constraint(model.A - L0 @ model.C, alpha)>=0:
        L0s.append(L0)
L0s = np.array(L0s)
print(f"Number of stable initial guesses: {L0s.shape[0]} out of {ntrials}")

all_dicts = []
for N in N_list:
    print(f" \n \n      ---- Solving for N = {N} ---- ")
    all_iterate_lists = []
    nice_flag = []
    for i, L0 in enumerate(L0s):
        print(f"Solving for initial guess {i+1}/{L0s.shape[0]}")
        _, stats = solve_PEM(model, ys[:N], L0, alpha)
        nice_flag.append(stats['success'])
        all_iterate_lists.append(stats['iterates'])

    dict_fixed_N = {
        'all_iterate_lists': all_iterate_lists,
        'nice_flag': nice_flag,
        'N':N
    }
    all_dicts.append(dict_fixed_N)

dict_complete = {
        'L1_range': L1_range,
        'L2_range': L2_range,
        'alpha': alpha,
        'ys': ys,
        'model': model,
        'all_dicts_fixed_N': all_dicts
    }
file_saving = join(main_dir, "savings", f"illustration_opt_data.pkl")
with open(file_saving, 'wb') as f:
    pickle.dump(dict_complete, f)
