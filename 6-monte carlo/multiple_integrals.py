import numpy as np


def rho(z):
    return 0.25 * z + 0.75


def random_coordinates():
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def monte_carlo_integration(N):
    for _ in range(N):
        x, y, z = random_coordinates()
        




