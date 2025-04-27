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

    return S, energies, magnetizations


# Ensemble average
ensemble_energies = []
ensemble_m = []

for _ in tqdm(range(n_ensemble)):
    S, energies, m = Ising_model(T=T, J=J, L=L, n_steps=n_steps, n_measure=n_measure)
    ensemble_energies.append(energies)
    ensemble_m.append(m)

avg_energies = np.mean(ensemble_energies, axis=0)
avg_magnetizations = np.mean(ensemble_m, axis=0)

plt.figure(figsize=(8, 6))
plt.imshow(S, cmap="RdYlBu", vmin=-1, vmax=1)
plt.colorbar(ticks=[-1, 1], label="Spin")
plt.title(f"Ising Model\nT={T} n_steps={n_steps} n_measure={n_measure}")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(avg_energies)
plt.xlabel("Measurement")
plt.ylabel("Energy")
plt.title(
    f"Energy Evolution\nT={T} n_steps={n_steps} n_measure={n_measure} n_ensemble={n_ensemble}"
)
plt.axhline(
    y=avg_energies[-1],
    color="r",
    linestyle="--",
    label=f"Final: {avg_energies[-1]:.2f}, Mean: {np.mean(avg_energies):.2f}",
)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(avg_magnetizations)
plt.xlabel("Measurement")
plt.ylabel("Magnetization")
plt.title(
    f"Magnetization Evolution\nT={T} n_steps={n_steps} n_measure={n_measure} n_ensemble={n_ensemble}"
)
plt.axhline(
    y=avg_magnetizations[-1],
    color="r",
    linestyle="--",
    label=f"Final: {avg_magnetizations[-1]:.2f}, Mean: {np.mean(avg_magnetizations):.2f}",
)
plt.legend()
plt.show()


# Magnetization vs T
T_range = np.linspace(1, 4.0, 20)
m_vs_T = []

for T in tqdm(T_range):
    ensemble_m = []
    for _ in range(n_ensemble):
        _, _, m = Ising_model(T=T, J=J, L=L, n_steps=n_steps, n_measure=n_measure)
        ensemble_m.append(np.mean(m[-10:]))
    m_vs_T.append(np.mean(ensemble_m))

plt.figure(figsize=(8, 6))
plt.plot(T_range, m_vs_T, "o-")
plt.xlabel("Temperature (T)")
plt.ylabel("Average Magnetization")
plt.title(
    f"m vs Temperature\nJ={J} L={L} n_steps={n_steps} n_measure={n_measure} n_ensemble={n_ensemble}"
)
plt.grid(True)
plt.show()
