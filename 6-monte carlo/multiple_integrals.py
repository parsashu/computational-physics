import numpy as np


def rho(r, theta):
    # return 0.25 * r * np.cos(theta) + 0.75
    return 1


def dm(r, theta):
    return r**2 * np.sin(theta) * rho(r, theta)


def r2_dm(r, theta):
    return r**2 * dm(r, theta)


def random_coordinates():
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    return r, theta, phi


def monte_carlo_integration(N):

    mean_m = 0
    mean_sqr_m = 0

    for _ in range(N):
        r, theta, phi = random_coordinates()

        mean_m += dm(r, theta)
        mean_sqr_m += (dm(r, theta)) ** 2

        mean_m /= N
        mean_sqr_m /= N

    I = mean_m * np.pi**2
    sigma = np.sqrt(mean_sqr_m - mean_m**2)
    delta = sigma / np.sqrt(N)

    return float(I), float(sigma), float(delta)
