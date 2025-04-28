import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 10
J = 1
N = L * L
n_steps = 1000000
n_measure = 1000
n_ensemble = 5


def delta_energy(S, i, j):
    return (
        2
        * S[i, j]
        * (
            S[(i + 1) % L, j]  # right neighbor
            + S[(i - 1) % L, j]  # left neighbor
            + S[i, (j + 1) % L]  # down neighbor
            + S[i, (j - 1) % L]  # up neighbor
        )
    )


def total_energy(S):
    energy = 0
    for i in range(L):
        for j in range(L):
            energy += (
                -J
                * S[i, j]
                * (
                    S[(i + 1) % L, j]  # right neighbor
                    + S[i, (j + 1) % L]  # down neighbor
                )
            )
    return energy


def magnetization(S):
    return abs(np.sum(S)) / N


def correlation(S, l):
    sum_s = 0
    sum_s2 = 0
    sum_shift_x = 0
    sum_shift_y = 0
    sum_prod_x = 0
    sum_prod_y = 0

    for i in range(L):
        for j in range(L):
            s = S[i, j]
            sx = S[(i + l) % L, j]
            sy = S[i, (j + l) % L]
            sum_s += s
            sum_s2 += s**2
            sum_shift_x += sx
            sum_shift_y += sy
            sum_prod_x += s * sx
            sum_prod_y += s * sy

    mean_s = sum_s / N
    mean_s2 = sum_s2 / N
    mean_shift_x = sum_shift_x / N
    mean_shift_y = sum_shift_y / N
    mean_prod_x = sum_prod_x / N
    mean_prod_y = sum_prod_y / N
    var_s = mean_s2 - mean_s**2

    # All spins are the same
    if var_s == 0:
        return 1.0

    Cx = mean_prod_x - mean_s * mean_shift_x
    Cy = mean_prod_y - mean_s * mean_shift_y
    return 0.5 * (Cx + Cy) / var_s


def Ising_model(T, J, L, n_steps, n_measure):
    """Metropolis algorithm for the Ising model"""
    S = np.random.choice([-1, 1], (L, L))
    boltzmann_factors = {dE: np.exp(-dE * J / T) for dE in [-8, -4, 0, 4, 8]}

    energies = []
    magnetizations = []
    correlations = []
    measure_interval = n_steps // n_measure

    # All random numbers and indices in the whole simulation
    indices = np.random.randint(0, L, size=(n_steps, 2))
    random_numbers = np.random.random(n_steps)

    for step in range(n_steps):
        i, j = indices[step]

        dE = delta_energy(S, i, j)
        if random_numbers[step] < boltzmann_factors[int(dE)]:
            S[i, j] = -S[i, j]

        if step % measure_interval == 0:
            energies.append(total_energy(S))
            magnetizations.append(magnetization(S))
            corr = [correlation(S, l) for l in range(L // 2)]
            correlations.append(corr)

    var_E = np.var(energies)
    var_m = np.var(magnetizations)
    mean_m = np.mean(magnetizations)
    mean_corr = np.mean(correlations, axis=0)

    Cv = var_E / (T**2)
    chi = var_m / T

    return Cv, chi, mean_m, mean_corr


# Ensemble average
T_range = np.linspace(1.5, 3.5, 15)

Cv_vs_T = []
chi_vs_T = []
m_vs_T = []
corr_vs_T = []

for T in tqdm(T_range):
    ensemble_Cv = []
    ensemble_chi = []
    ensemble_m = []
    ensemble_corr = []

    for _ in range(n_ensemble):
        Cv, chi, mean_m, mean_corr = Ising_model(
            T=T, J=J, L=L, n_steps=n_steps, n_measure=n_measure
        )
        ensemble_Cv.append(Cv)
        ensemble_chi.append(chi)
        ensemble_m.append(mean_m)
        ensemble_corr.append(mean_corr)

    m_vs_T.append(np.mean(ensemble_m))
    Cv_vs_T.append(np.mean(ensemble_Cv))
    chi_vs_T.append(np.mean(ensemble_chi))
    corr_vs_T.append(np.mean(ensemble_corr, axis=0))


plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(T_range, Cv_vs_T, "o-")
plt.xlabel("Temperature (T)")
plt.ylabel("Heat Capacity (Cv)")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(T_range, chi_vs_T, "o-")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetic Susceptibility (χ)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(T_range, m_vs_T, "o-")
plt.xlabel("Temperature (T)")
plt.ylabel("Average Magnetization")
plt.grid(True)

plt.suptitle(
    f"Thermodynamic Properties vs Temperature\nJ={J} L={L} n_steps={n_steps} n_measure={n_measure} n_ensemble={n_ensemble}"
)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
l_range = np.arange(L // 2)
plt.plot(l_range, corr_vs_T, "o-", label=f"T = {T:.2f}")
plt.xlabel("Distance (l)")
plt.ylabel("Correlation C(l)")
plt.title("Spin-Spin Correlation Function")
plt.grid(True)
plt.legend()
plt.show()
