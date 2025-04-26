import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 200
T = 0.1
J = 1
N = L * L
n_steps = 10000000
n_measure = 100000

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


S, energies, magnetizations = metropolis(n_steps=n_steps, n_measure=n_measure)

plt.figure(figsize=(8, 8))
plt.imshow(S, cmap="RdYlBu", vmin=-1, vmax=1)
plt.colorbar(ticks=[-1, 1], label="Spin")
plt.title(f"Ising Model\nT={T} n_steps={n_steps} n_measure={n_measure}")
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(energies)
plt.xlabel("Measurement")
plt.ylabel("Energy")
plt.title("Energy Evolution")
plt.axhline(
    y=energies[-1],
    color="r",
    linestyle="--",
    label=f"Final: {energies[-1]:.2f}, Mean: {np.mean(energies):.2f}",
)
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(magnetizations)
plt.xlabel("Measurement")
plt.ylabel("Magnetization")
plt.title("Magnetization Evolution")
plt.axhline(
    y=magnetizations[-1],
    color="r",
    linestyle="--",
    label=f"Final: {magnetizations[-1]:.2f}, Mean: {np.mean(magnetizations):.2f}",
)
plt.legend()
plt.show()
