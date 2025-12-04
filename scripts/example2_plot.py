import numpy as np
import pickle
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from os.path import dirname, join
main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)
from LTI import constraint
from latexify import latexify
latexify()

def rho(A):
    return max(abs(np.linalg.eigvals(A)))

ngrid = 200
def first_processing(dict_complete, ngrid):
    model = dict_complete['model']
    L_star, S_star = model.KalmanGain()
    L_symbol = ca.SX.sym("L", (model.nx, model.ny))
    E_fn = ca.Function("E", [L_symbol], [model.E(L_symbol, dict_complete['ys'])])
    L1_grid = np.linspace(*dict_complete['L1_range'], ngrid)
    L2_grid = np.linspace(*dict_complete['L2_range'], ngrid)
    L1_mesh, L2_mesh = np.meshgrid(L1_grid, L2_grid)
    rho_mesh = np.empty((ngrid, ngrid))
    h_mesh = -1 * np.ones((ngrid, ngrid))
    L_mesh = np.empty((ngrid, ngrid, model.nx, model.ny))
    for i in range(ngrid):
        for j in range(ngrid):
            L_mesh[i,j] = np.array([L1_mesh[i, j], L2_mesh[i, j]])[:, None]
            AmLC = model.A - L_mesh[i, j] @ model.C
            rho_mesh[i, j] = abs(np.linalg.eigvals(AmLC)).max()
            h_mesh[i, j] = constraint(AmLC, dict_complete['alpha'])

    dict_cst = {
        "Lstar": L_star,
        "E_fn": E_fn,
        "h_mesh": h_mesh,
        "rho_mesh": rho_mesh,
        "L_mesh": L_mesh,
    }
    return dict_cst

def plot_on_ax(ax, dict_to_plot, dict_cst, xlabel=True, ylabel=True):
    h_mesh = dict_cst["h_mesh"]
    rho_mesh = dict_cst["rho_mesh"]
    L_mesh = dict_cst["L_mesh"]
    L1_mesh = L_mesh[:,:,0,0]
    L2_mesh = L_mesh[:,:,1,0]
    ngrid = h_mesh.shape[0]
    L1_min, L1_max = L1_mesh.min(), L1_mesh.max()
    L2_min, L2_max = L2_mesh.min(), L2_mesh.max()

    all_iterate_lists = dict_to_plot['all_iterate_lists']
    nice_flag = dict_to_plot['nice_flag']
    N = dict_to_plot['N']

    f_mesh = np.inf * np.ones((ngrid, ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            if h_mesh[i, j] > 0:
                e = dict_cst["E_fn"](L_mesh[i, j]).full()[:N, :]
                f_mesh[i, j] = np.sum(e**2)
    i_sol = np.argmin(f_mesh.flatten())
    L1_sol = L1_mesh.flatten()[i_sol]
    L2_sol = L2_mesh.flatten()[i_sol]

    # plot the feasible sets rho(L) < 1 and h(L) > 0 and f(L) < f(Lstar)
    color_rho = 'orange'
    ax.contourf(L1_mesh, L2_mesh, rho_mesh, levels=[0., 1.], alpha=0.1, colors=[color_rho]) # region
    ax.contour(L1_mesh, L2_mesh, rho_mesh, levels=[1.], linestyles='--', colors=[color_rho]) # boundary
    for_legend_region_rho = (
        mpatches.Patch(alpha=0.4, facecolor=color_rho, edgecolor=color_rho, linewidth=1, linestyle='--'),
        r"$\rho(A-LC) \leq 1$"
    )

    color_h = 'purple'
    ax.contourf(L1_mesh, L2_mesh, h_mesh, levels=[0, np.max(h_mesh)], alpha=0.1, colors=[color_h])
    ax.contour(L1_mesh, L2_mesh, h_mesh, levels=[0], linestyles='--', colors=[color_h])
    for_legend_region_h = (
        mpatches.Patch(alpha=0.4, facecolor=color_h, edgecolor=color_h, linewidth=1, linestyle='--'),
        r"$\alpha \mathrm{Tr}(P(L)-I) \leq 1$"
    )
    ax.plot(L1_sol, L2_sol, "o", markersize=15, color="green", alpha=0.5, label="Global minimizer")
    label1 = "Initial guesses"
    label2 = "Solutions found"
    for i, xs in enumerate(all_iterate_lists):
        if nice_flag[i]:
            color = 'blue'
        else:
            color = 'red'
        ax.plot(xs[0,0], xs[0,1], "o", alpha=0.2, markersize=5, color=color, label=label1) # initial guess
        ax.plot(xs[:,0], xs[:,1], "-", alpha=0.08, color=color) # iterates
        ax.plot(xs[-1,0], xs[-1,1], "x", markersize=10, color=color, label=label2) # solution
        label1, label2 = None, None
    ax.plot([], [], "-", alpha=0.2, color=color, label="Iterates") # global solution again for legend
    ax.plot(dict_cst["Lstar"][0], dict_cst["Lstar"][1], "*", color="red", markersize=8, label=r"True Kalman gain")
    if xlabel:
        ax.set_xlabel(r"$L_{11}$")
    if ylabel:
        ax.set_ylabel(r"$L_{21}$")
    ax.set_xlim(L1_min, L1_max)
    ax.set_ylim(L2_min, L2_max)
    ax.set_title(r"$N="+str(N)+r"$")

    additional_handles_and_labels = [for_legend_region_rho, for_legend_region_h]
    return ax, additional_handles_and_labels

file_saving = join(main_dir, "savings", f"illustration_opt_data.pkl")
with open(file_saving, 'rb') as f:
    dict_complete = pickle.load(f)
dict_cst = first_processing(dict_complete, ngrid)
all_dicts_fixed_N = dict_complete['all_dicts_fixed_N']

n_plots = len(all_dicts_fixed_N)
# Now this is specific to the case of 4 plots
fig, axs = plt.subplots(3, 2, figsize=(6, 6),
                        gridspec_kw={'height_ratios':[1, 1, 0.2]})
plt.subplots_adjust(
    wspace=0.15,  # width space between columns
    hspace=0.45   # height space between rows
)
_, additional_handles = plot_on_ax(axs[0, 0], all_dicts_fixed_N[0], dict_cst, xlabel=False)
plot_on_ax(axs[0, 1], all_dicts_fixed_N[1], dict_cst, xlabel=False, ylabel=False)
plot_on_ax(axs[1, 0], all_dicts_fixed_N[2], dict_cst)
plot_on_ax(axs[1, 1], all_dicts_fixed_N[3], dict_cst, ylabel=False)

axs[2, 0].axis("off")
axs[2, 1].axis("off")
h, l = axs[0,0].get_legend_handles_labels()
h = h + [item[0] for item in additional_handles]
l = l + [item[1] for item in additional_handles]

axs[2, 0].legend(h, l, loc="center left", ncol=3,
                    bbox_to_anchor=(-0.12, 0),
                    columnspacing=0.5
                    )
IMAGES_dir = join(main_dir, "images")
fig.savefig(f"{IMAGES_dir}\opti.pdf", bbox_inches='tight', pad_inches=0.01) # save pdf
plt.show(block=False)
plt.pause(10) # brief pause to ensure plot renders correctly


