import numpy as np
import matplotlib.pyplot as plt


def init():
    global N, L, T, J, boltzmann_factors
    L = 10
    T = 1
    J = 1 / T
    N = L * L
    S0 = np.random.choice([-1, 1], (L, L))
    boltzmann_factors = {
        8: np.float64(0.00033546262790251185),
        4: np.float64(0.01831563888873418),
        0: np.float64(1.0),
        -4: np.float64(54.598150033144236),
        -8: np.float64(2980.9579870417283),
    }
    return S0


def delta_energy(S, i, j):
    return (
        -2
        * S[i, j]
        * (
            S[(i + 1) % L, j]
            + S[(i - 1) % L, j]
            + S[i, (j + 1) % L]
            + S[i, (j - 1) % L]
        )
    )


def metropolis(S):
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)

    S_new = S.copy()
    S_new[i, j] = -S[i, j]
    dE = delta_energy(S, i, j)
    boltzmann_factor = boltzmann_factors[int(dE)]
    if np.random.rand() < boltzmann_factor:
        S = S_new
    return S


S0 = init()
S = metropolis(S0)

plt.figure(figsize=(8, 8))
plt.imshow(S, cmap='RdYlBu', vmin=-1, vmax=1)
plt.colorbar(ticks=[-1, 1], label='Spin')
plt.title('Ising Model State')
plt.show()


