import numpy as np
import matplotlib.pyplot as plt

R = 1
C = 1
tau = R * C
Q0 = 10

T = 8
h = 0.01
n_steps = int(T / h)

t = np.linspace(0, T, n_steps)
Q = np.zeros(n_steps)

Q[0] = Q0
Q[1] = Q0 * (1 - h / tau)

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
