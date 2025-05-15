import numpy as np
import matplotlib.pyplot as plt

R = 1e3
C = 1e-6
tau = R * C

T = 0.01
h = 0.0001
n_steps = int(T / h)

t = np.linspace(0, T, n_steps)
Q = np.zeros(n_steps)

Q[0] = 5
Q[1] = 5 * (1 - h / tau)

for n in range(1, n_steps - 1):
    f_n = -Q[n] / tau
    Q[n + 1] = Q[n - 1] + 2 * f_n * h

plt.plot(t, Q, label="RC Charge")
plt.xlabel("Time (s)")
plt.ylabel("Charge (C)")
plt.title(f"RC Charge using new algorithm (h = {h}s)")
plt.legend()
plt.grid(True)
plt.show()
