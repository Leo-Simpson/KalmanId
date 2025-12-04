import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
from os.path import join, dirname
main_dir = dirname(dirname(__file__))
src_dir = join(main_dir, 'src')
sys.path.append(src_dir)

from latexify import latexify
latexify()

figsize=(5,2.5)
def create_plot(Nlist, all_errors):
    fig, ax = plt.subplots(figsize=figsize)
    label=r"$\Vert L^\star - \hat{L}_N \Vert_{\infty}$"
    for errors in all_errors:
        ax.plot(Nlist, errors, 'x', label=label, color="blue", alpha=0.9)
        label=None
    mean_errors = np.mean(all_errors, axis=0)
    c = np.mean( mean_errors * np.sqrt(Nlist) )
    ax.plot(Nlist, c /np.sqrt(Nlist), label=r"$\frac{c}{\sqrt{N}}$", color="black", linewidth=2)

    min_error = np.min(all_errors, axis=0)
    max_error = np.max(all_errors, axis=0)
    ax.fill_between(Nlist, min_error, max_error, color="blue", alpha=0.1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Number of data points $N$")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig, ax

file_saving = join(main_dir, "savings", "consistency_data.pkl")
with open(file_saving, 'rb') as f:
    loaded = pickle.load(f)

fig, ax = create_plot( np.array(loaded["Nlist"]), loaded["all_errors"])
IMAGES_dir = join(main_dir, "images")
fig.savefig(f"{IMAGES_dir}\consistency.pdf", bbox_inches='tight', pad_inches=0.01) # save pdf
plt.show(block=False)
plt.pause(10) # brief pause to ensure plot renders correctly
