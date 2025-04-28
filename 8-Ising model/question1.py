import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from ising import Ising_model

L = 200
T = 5
J = 1
N = L * L
n_steps = 100000
n_measure = 10000
n_ensemble = 1

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
