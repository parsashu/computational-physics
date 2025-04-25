import numpy as np
import matplotlib.pyplot as plt

L = 10
T = 1000
J = 1.0
N = L * L
n_steps = 1000000
n_measure = 1000

S0 = np.random.choice([-1, 1], (L, L))
boltzmann_factors = {dE: np.exp(-dE / T) for dE in [-8, -4, 0, 4, 8]}


def delta_energy(S, i, j):
    return (
        -2
        * J
        * S[i, j]
        * (
            S[(i + 1) % L, j]
            + S[(i - 1) % L, j]
            + S[i, (j + 1) % L]
            + S[i, (j - 1) % L]
        )
    )


def total_energy(S):
    energy = 0
    for i in range(L):
        for j in range(L):
            energy += -J * S[i, j] * (S[(i + 1) % L, j] + S[i, (j + 1) % L])
    return energy


def magnetization(S):
    return abs(np.sum(S)) / N


def metropolis(S, n_steps=1000, n_measure=1000):
    energies = []
    magnetizations = []

    for step in range(n_steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        dE = delta_energy(S, i, j)
        if np.random.rand() < boltzmann_factors[int(dE)]:
            S[i, j] = -S[i, j]

        if step % (n_steps // n_measure) == 0:
            energies.append(total_energy(S))
            magnetizations.append(magnetization(S))

    return S, energies, magnetizations


S, energies, magnetizations = metropolis(S0, n_steps=n_steps, n_measure=n_measure)

fig1 = plt.figure(figsize=(8, 8))
plt.imshow(S, cmap="RdYlBu", vmin=-1, vmax=1)
plt.colorbar(ticks=[-1, 1], label="Spin")
plt.title(f"Spin Configuration\nT={T}")
plt.show()

fig2 = plt.figure(figsize=(8, 8))
plt.plot(energies)
plt.xlabel("Measurement")
plt.ylabel("Energy")
plt.title("Energy Evolution")
plt.show()

fig3 = plt.figure(figsize=(8, 8))
plt.plot(magnetizations)
plt.xlabel("Measurement")
plt.ylabel("Magnetization")
plt.title("Magnetization Evolution")
plt.show()

print(f"Final energy: {energies[-1]}")
print(f"Final magnetization: {magnetizations[-1]}")
print(f"Average energy: {np.mean(energies)}")
print(f"Average magnetization: {np.mean(magnetizations)}")
