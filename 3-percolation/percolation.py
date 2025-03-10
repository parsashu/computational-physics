# 3-percolation/percolation.py
import matplotlib.pyplot as plt
import numpy as np

L = 10

def plot_grid(L):
    grid = np.zeros((L, L))

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-0.5, L, 1))
    ax.set_yticks(np.arange(-0.5, L, 1))
    ax.grid(True, which='both', color='black', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(grid, cmap='Greys', extent=[-0.5, L-0.5, -0.5, L-0.5])

    plt.show()

# Example usage:
plot_grid(L)