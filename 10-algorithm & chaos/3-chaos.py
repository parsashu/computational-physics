import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def logistic_map(x0, r, iterations=1000):
    x = np.zeros(iterations)
    x[0] = x0
    for i in range(1, iterations):
        x[i] = 4 * r * x[i - 1] * (1 - x[i - 1])
    return x


r = np.linspace(0.6, 0.893, 10000)
x0 = 0.65

for r_ in tqdm(r):
    x = logistic_map(x0, r_)
    x_samples = x[-100:]
    x_unique = np.unique(x_samples)
    plt.plot([r_] * len(x_unique), x_unique, "k.", markersize=0.5)

plt.show()
