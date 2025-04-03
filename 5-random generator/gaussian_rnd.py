from random import random
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

N = 100000


def gaussian_rnd(sigma):
    x1 = random()
    x2 = random()

    if x1 >= 0.9999999:
        x1 = 0.9999999

    rho = math.sqrt(-2 * (sigma**2) * math.log(1 - x1))
    theta = 2 * math.pi * x2

    y1 = rho * math.cos(theta)
    y2 = rho * math.sin(theta)
    return y1, y2


numbers = []
for _ in range(N // 2):
    y1, y2 = gaussian_rnd(1)
    numbers.extend([y1, y2])

plt.figure(figsize=(10, 6))
hist, bin_edges = np.histogram(numbers, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

mu, sigma = stats.norm.fit(numbers)
x = np.linspace(min(numbers), max(numbers), 100)
gaussian_curve = stats.norm.pdf(x, mu, sigma)

plt.hist(numbers, bins=50, rwidth=0.8, density=True, alpha=0.7, label='Histogram')
plt.plot(x, gaussian_curve, 'r-', linewidth=2, label=f'Gaussian Fit: μ={mu:.2f}, σ={sigma:.2f}')
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title(f"Distribution of {N} Gaussian Random Numbers with Fitted Curve")
plt.grid(axis="y", alpha=0.75)
plt.legend()
plt.show()
