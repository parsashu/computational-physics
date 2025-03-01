import numpy as np
import matplotlib.pyplot as plt


def julia_set(c, max_iter):
    w = 800
    h = 800
    r = 1.5
    radius = 2

    # Create a complex grid of coordinates
    y, x = np.ogrid[-r : r : h * 1j, -r : r : w * 1j]
    z = x + y * 1j

    # Initialize iteration count array
    iteration = np.zeros(z.shape, dtype=int)

    # True if the point hasn't escaped
    mask = np.ones(z.shape, dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c

        # Update mask for points that escape
        mask_new = np.abs(z) <= radius

        # Record iteration count for points that just escaped at this step
        iteration[mask & ~mask_new] = i
        mask = mask_new

    # Set max iteration for points that never escape
    iteration[mask] = max_iter
    return iteration


def plot_julia(c, cmap="magma", max_iter=350):
    julia = julia_set(c, max_iter)

    # Create a custom colormap to make the set itself black
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Plot the Julia set
    plt.figure(figsize=(10, 10))
    plt.imshow(julia, cmap=cmap)
    plt.title(f"Julia Set for c = {c}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Book
plot_julia(c=-1j)
# plot_julia(c=-0.4 - 0.6j)
# plot_julia(c=-0.4 + 0.6j)
# plot_julia(c=-0.12 - 0.75j, max_iter=20)
# plot_julia(c=-0.6, max_iter=20)
# plot_julia(c=-0.8 + 0.16j)

# Internet
# plot_julia(c=-0.7 + 0.27j, max_iter=100, cmap="hot")
# plot_julia(c=0.285 + 0.01j, max_iter=100, cmap="hot")
