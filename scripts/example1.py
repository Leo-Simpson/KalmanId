import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import sys
from os.path import join, dirname
main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)
from Models import model_1state

# Latexfiy
from latexify import latexify
latexify()

rng = np.random.default_rng(42)
a = 0.9
S_star = 1.
Lstar = 0.8
model = model_1state(a, Lstar, S=S_star)

def Vbar(L, Lstar):
    return  S_star * (1 + (L - Lstar)**2 / (1 - (a - L)**2))

def local_minimizers(V):
    mins = []
    for i in range(1, len(V)-1):
        if V[i-1] >= V[i] and V[i+1] >= V[i]:
            mins.append(i)
    return mins

L = np.linspace(-0.5, 2, 1000)
feasible_mask = (a - L > -1.) & (a - L < 1.)
Lfeasible = L[feasible_mask]

Vbars = Vbar(Lfeasible, Lstar)
Ns = [7, 15, 500] # , 2000 for one more accurate
ws = rng.normal(0, 1, size=(Ns[-1],model.nw))
ys = model.simulate(ws)

# Evaluate E(L) for all L
L_symbol = ca.SX.sym("L", (model.nx, model.ny))
E_fn = ca.Function("V", [L_symbol], [model.E(L_symbol, ys)])
E_eval = np.array([E_fn(l).full() for l in L])

Vsim = []
for N in Ns:
    V_eval = np.sum(E_eval[:, :N, :]**2, axis=(1,2)) / N
    Vsim.append(V_eval)

# Make the plot
figsize = (5.5, 2.8)
min_V, max_V = 0.5 * S_star, 2.5 * S_star

fig, axs = plt.subplots(2,1, figsize=figsize,
    gridspec_kw={'height_ratios':[3, 1]}
)
ax = axs[0]
ax.set_xlabel(r"Gain $L$")
ax.set_ylabel(r"Objective value")

VN_label = r"$V_N(L)$, N="+f"{Ns[0]}, {Ns[1]}, {Ns[2]}"
for i in range(len(Ns)):
    ax.plot(L, Vsim[i], "-",  alpha=0.3, color="blue", label=VN_label)
    VN_label = None  # only label the first one

# plot local minimizers
loc_min_label = "loc. min. of $V_N(L)$"
for i in range(len(Ns)):
    k_local_mins = local_minimizers(Vsim[i])
    for k in k_local_mins:
        ax.plot(L[k], Vsim[i][k], "+", markersize=6, label=loc_min_label, color="blue")
        loc_min_label = None  # only label the first one

# Plot Vbar
ax.plot(Lfeasible, Vbars, "-.", label=r"limit $\bar{V}(L)$",  alpha=0.8, color="purple")

# Plot the global minimum of Vbar
ax.axvline(Lstar, color='green', linestyle='dashed', label=r"$L^\star$ \& $\bar{V}(L^\star)$")
ax.axhline(Vbar(Lstar, Lstar), color='green', linestyle='dashed')

# Shade unstable regions
ax.axvspan(L.min(), Lfeasible.min(), alpha=0.2, color="red", label='unstable region')
ax.axvspan(Lfeasible.max(), L.max(), alpha=0.2, color="red")

# Set limits
ax.set_ylim([min_V, max_V])
ax.set_xlim([L.min(), L.max()])

ax4legend = axs[1]
ax4legend.axis("off")
h, l = ax.get_legend_handles_labels()
ax4legend.legend(h, l, loc="center left", ncol=3,
                    bbox_to_anchor=(-0.05, 0),
                    columnspacing=0.8
                    )
IMAGES_dir = join(main_dir, "images")
fig.savefig(f"{IMAGES_dir}\illustration.pdf", bbox_inches='tight', pad_inches=0.01) # save pdf

plt.show(block=False)
plt.pause(10) # brief pause to ensure plot renders correctly
