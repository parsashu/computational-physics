import numpy as np
from coloring import coloring, is_connected_to_infinite_cluster_color
from hoshen_kopelman import (
    hoshen_kopelman,
    is_connected_to_infinite_cluster_hoshen,
    correlation_length,
)
import matplotlib.pyplot as plt

length = 10
p = 0
Q_color = []
Q_hoshen = []

Q_inf_color = []
Q_inf_hoshen = []

Xi_hoshen = []

while p <= 1:
    n_percolate_color = 0
    n_percolate_hoshen = 0

    n_connect_color = 0
    n_connect_hoshen = 0

    correlation_lengths = []

    for _ in range(100):
        random_values = np.random.random((length, length))

        # grid_color, percolates_color = coloring(length, random_values, p)
        grid_hoshen, percolates_hoshen = hoshen_kopelman(length, random_values, p)

        correlation_lengths.append(correlation_length(grid_hoshen))

    #     if percolates_color:
    #         n_percolate_color += 1
    #     if percolates_hoshen:
    #         n_percolate_hoshen += 1

    #     if is_connected_to_infinite_cluster_color(grid_color):
    #         n_connect_color += 1
    #     if is_connected_to_infinite_cluster_hoshen(grid_hoshen):
    #         n_connect_hoshen += 1

    # Q_color.append(n_percolate_color / 100)
    # Q_hoshen.append(n_percolate_hoshen / 100)

    # Q_inf_color.append(n_connect_color / 100)
    # Q_inf_hoshen.append(n_connect_hoshen / 100)

    Xi_hoshen.append(np.mean(correlation_lengths))

    p += 0.05


p_values = [i * 0.05 for i in range(20)]

# plt.figure(figsize=(10, 6))
# plt.plot(p_values, Q_color, "o-", label="Coloring Algorithm")
# plt.plot(p_values, Q_hoshen, "s-", label="Hoshen-Kopelman Algorithm")
# plt.xlabel("Occupation Probability (p)")
# plt.ylabel("Percolation Probability (q)")
# plt.title(f"Percolation Probability vs Occupation Probability (L={length})")
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(p_values, Q_inf_color, "o-", label="Coloring Algorithm")
# plt.plot(p_values, Q_inf_hoshen, "s-", label="Hoshen-Kopelman Algorithm")
# plt.xlabel("Occupation Probability (p)")
# plt.ylabel("Infinite Cluster Probability (q)")
# plt.title(f"Infinite Cluster Probability vs Occupation Probability (L={length})")
# plt.legend()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(p_values, Xi_hoshen, "o-", label="Hoshen-Kopelman Algorithm")
plt.xlabel("Occupation Probability (p)")
plt.ylabel("Correlation Length (Xi)")
plt.title(f"Correlation Length vs Occupation Probability (L={length})")
plt.legend()
plt.show()
