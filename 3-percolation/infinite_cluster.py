import numpy as np
from coloring import coloring
from hoshen_kopelman import hoshen_kopelman
import matplotlib.pyplot as plt

length = 10
p = 0
q_color = []
q_hoshen = []



while p <= 1:
    n_percolate_color = 0
    n_percolate_hoshen = 0

    for _ in range(100):
        random_values = np.random.random((length, length))
        percolates_color = coloring(length, random_values, p)
        percolates_hoshen = hoshen_kopelman(length, random_values, p)

        if percolates_color:
            n_percolate_color += 1
        if percolates_hoshen:
            n_percolate_hoshen += 1

    q_color.append(n_percolate_color / 100)
    q_hoshen.append(n_percolate_hoshen / 100)
    p += 0.05


p_values = [i * 0.05 for i in range(20)]

plt.figure(figsize=(10, 6))
plt.plot(p_values, q_color, "o-", label="Coloring Algorithm")
plt.plot(p_values, q_hoshen, "s-", label="Hoshen-Kopelman Algorithm")
plt.xlabel("Occupation Probability (p)")
plt.ylabel("Percolation Probability (q)")
plt.title("Percolation Probability vs Occupation Probability")
plt.legend()
plt.show()
