import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 10
T = 5
J = 1
N = L * L
n_steps = 10000
n_measure = 1000
n_ensemble = 100


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


def Ising_model(T, J, L, n_steps, n_measure):
    """Metropolis algorithm for the Ising model"""
    S = np.random.choice([-1, 1], (L, L))
    boltzmann_factors = {dE: np.exp(-dE * J / T) for dE in [-8, -4, 0, 4, 8]}

    energies = []
    magnetizations = []
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

    var_E = np.var(energies)
    var_m = np.var(magnetizations)
    mean_m = np.mean(magnetizations)

    Cv = var_E / (T**2)
    chi = var_m / T

    return Cv, chi, mean_m


# Ensemble average
T_range = np.linspace(1, 4.0, 20)

Cv_vs_T = []
chi_vs_T = []
m_vs_T = []

for T in tqdm(T_range):
    ensemble_Cv = []
    ensemble_chi = []
    ensemble_m = []

    for _ in range(n_ensemble):
        Cv, chi, mean_m = Ising_model(
            T=T, J=J, L=L, n_steps=n_steps, n_measure=n_measure
        )
        ensemble_Cv.append(Cv)
        ensemble_chi.append(chi)
        ensemble_m.append(mean_m)

    m_vs_T.append(np.mean(ensemble_m))
    Cv_vs_T.append(np.mean(ensemble_Cv))
    chi_vs_T.append(np.mean(ensemble_chi))


plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(T_range, Cv_vs_T, "o-")
plt.ylabel("Heat Capacity (Cv)")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(T_range, chi_vs_T, "o-")
plt.ylabel("Magnetic Susceptibility (χ)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(T_range, m_vs_T, "o-")
plt.xlabel("Temperature (T)")
plt.ylabel("Average Magnetization")
plt.grid(True)

plt.suptitle(
    f"Thermodynamic Properties vs Temperature\nJ={J} L={L} n_steps={n_steps} n_measure={n_measure} n_ensemble={n_ensemble}"
)
plt.tight_layout()
plt.show()
