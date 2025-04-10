import numpy as np


def rho(r, theta):
    return 0.25 * r * np.cos(theta) + 0.75


def dm(r, theta):
    return r**2 * np.sin(theta) * rho(r, theta)


def z_dm(r, theta):
    return r * np.cos(theta) * dm(r, theta)


def random_coordinates():
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, np.pi)
    return r, theta


def Z_cm(N):
    mean_m = 0
    mean_z = 0
    mean_sqr_m = 0
    mean_sqr_z = 0

    for _ in range(N):
        r, theta = random_coordinates()

        mean_m += dm(r, theta)
        mean_z += z_dm(r, theta)
        mean_sqr_m += (dm(r, theta)) ** 2
        mean_sqr_z += (z_dm(r, theta)) ** 2

    mean_m /= N
    mean_z /= N
    mean_sqr_m /= N
    mean_sqr_z /= N

    M = mean_m * 2 * np.pi**2
    sigma_m = np.sqrt(mean_sqr_m - mean_m**2)

    Z = mean_z * 2 * np.pi**2
    sigma_z = np.sqrt(mean_sqr_z - mean_z**2)

    Z_cm = Z / M
    sigma_Z_cm = abs(Z_cm) * np.sqrt((sigma_z / Z) ** 2 + (sigma_m / M) ** 2)
    delta_Z_cm = sigma_Z_cm / np.sqrt(N)

    return float(Z_cm), float(delta_Z_cm)


Z_cm_value, Z_cm_error = Z_cm(1000000)
print(f"Z_cm = {Z_cm_value:.6f} ± {Z_cm_error:.6f}")
