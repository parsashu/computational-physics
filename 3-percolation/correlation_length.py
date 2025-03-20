import numpy as np
from hoshen_kopelman import hoshen_kopelman, correlation_length
import matplotlib.pyplot as plt

length = 10
p = 0
n_runs = 100
Xi_hoshen = []
p_values = []

while p < 0.48:
    delta_p = 0.02
    correlation_lengths = []

    for _ in range(n_runs):
        random_values = np.random.random((length, length))
        grid_hoshen, percolates_hoshen = hoshen_kopelman(length, random_values, p)
        correlation_lengths.append(correlation_length(grid_hoshen))

    Xi_hoshen.append(np.mean(correlation_lengths))
    p_values.append(p)
    p += delta_p

while p < 0.68:
    delta_p = 0.005
    correlation_lengths = []

    for _ in range(n_runs):
        random_values = np.random.random((length, length))
        grid_hoshen, percolates_hoshen = hoshen_kopelman(length, random_values, p)
        correlation_lengths.append(correlation_length(grid_hoshen))

    Xi_hoshen.append(np.mean(correlation_lengths))
    p_values.append(p)
    p += delta_p

while p <= 1:
    delta_p = 0.02
    correlation_lengths = []

    for _ in range(n_runs):
        random_values = np.random.random((length, length))
        grid_hoshen, percolates_hoshen = hoshen_kopelman(length, random_values, p)
        correlation_lengths.append(correlation_length(grid_hoshen))

    Xi_hoshen.append(np.mean(correlation_lengths))
    p_values.append(p)
    p += delta_p

plt.figure(figsize=(10, 6))
plt.plot(p_values, Xi_hoshen, "o", label="Hoshen-Kopelman Algorithm")
plt.xlabel("Occupation Probability (p)")
plt.ylabel("Correlation Length (Xi)")
plt.title(f"Correlation Length vs Occupation Probability (L={length})")
plt.grid(True)
plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(min(Xi_hoshen), max(Xi_hoshen), 10))
plt.legend()
plt.show()