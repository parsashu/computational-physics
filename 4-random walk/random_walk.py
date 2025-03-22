"""
This script simulates a one-dimensional symmetric random walk and visualizes the path
taken by a point starting at the origin and moving in random directions in 1D space.

A random walk is a mathematical object that describes a path consisting of a succession
of random steps. In this 1D version, the point can move left or right at each step.
"""

import numpy as np
import matplotlib.pyplot as plt


def random_walk(num_steps):
    position = 0
    path = [0]

    for _ in range(num_steps):
        step = np.random.choice([-1, 1])
        position += step
        path.append(position)

    return path


num_steps = 1000

path = random_walk(num_steps)

plt.figure(figsize=(10, 6))
plt.plot(range(num_steps + 1), path, marker="o", markersize=3)
plt.title("One-dimensional Symmetric Random Walk")
plt.xlabel("Step Number")
plt.ylabel("Position")
plt.grid(True)
plt.show()
