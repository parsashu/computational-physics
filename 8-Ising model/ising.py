import numpy as np


def delta_energy(S, i, j):
    L = len(S)
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


def total_energy(S, J):
    L = len(S)
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
    N = len(S) ** 2
    return abs(np.sum(S)) / N


def Ising_model(T, J, L, n_steps, n_measure=None, mesure=True):
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

        if step % measure_interval == 0 and mesure:
            energies.append(total_energy(S, J))
            magnetizations.append(magnetization(S))

    return S, energies, magnetizations
