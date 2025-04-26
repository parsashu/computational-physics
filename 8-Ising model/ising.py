import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 200
T = 5
J = 1
N = L * L
n_steps = 1000000
n_measure = 10000
n_ensemble = 1

S = np.random.choice([-1, 1], (L, L))
right = np.roll(S, -1, axis=0)
left = np.roll(S, 1, axis=0)
up = np.roll(S, -1, axis=1)
down = np.roll(S, 1, axis=1)

boltzmann_factors = {dE: np.exp(-dE / T) for dE in [-8 * J, -4 * J, 0, 4 * J, 8 * J]}


def delta_energy(i, j):
    return 2 * J * S[i, j] * (right[i, j] + left[i, j] + up[i, j] + down[i, j])


def total_energy():
    return -J * np.sum(S * (right + up))


def magnetization():
    return abs(np.sum(S)) / N


def metropolis(n_steps, n_measure):
    """Metropolis algorithm for the Ising model"""
    energies = []
    magnetizations = []
    measure_interval = n_steps // n_measure

    # All random numbers and indices in the whole simulation
    indices = np.random.randint(0, L, size=(n_steps, 2))
    random_numbers = np.random.random(n_steps)

    for step in tqdm(range(n_steps)):
        i, j = indices[step]

        dE = delta_energy(i, j)
        if random_numbers[step] < boltzmann_factors[int(dE)]:
            S[i, j] = -S[i, j]
            # Update neighbors
            right[i, j] = S[(i + 1) % L, j]
            left[i, j] = S[(i - 1) % L, j]
            up[i, j] = S[i, (j + 1) % L]
            down[i, j] = S[i, (j - 1) % L]

        if step % measure_interval == 0:
            energies.append(total_energy())
            magnetizations.append(magnetization())

    return S, energies, magnetizations


# Ensemble average
ensemble_energies = []
ensemble_magnetizations = []

for _ in range(n_ensemble):
    S, energies, magnetizations = metropolis(n_steps=n_steps, n_measure=n_measure)
    ensemble_energies.append(energies)
    ensemble_magnetizations.append(magnetizations)

avg_energies = np.mean(ensemble_energies, axis=0)
avg_magnetizations = np.mean(ensemble_magnetizations, axis=0)

plt.figure(figsize=(8, 8))
plt.imshow(S, cmap="RdYlBu", vmin=-1, vmax=1)
plt.colorbar(ticks=[-1, 1], label="Spin")
plt.title(f"Ising Model\nT={T} n_steps={n_steps} n_measure={n_measure}")
plt.show()

plt.figure(figsize=(8, 8))
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

plt.figure(figsize=(8, 8))
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
