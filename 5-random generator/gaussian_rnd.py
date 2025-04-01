from random import random
import math
from matplotlib import pyplot as plt

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
plt.hist(numbers, bins=50, rwidth=0.8)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title(f"Distribution of {N} Gaussian Random Numbers")
plt.grid(axis="y", alpha=0.75)
plt.show()
