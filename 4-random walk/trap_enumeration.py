import numpy as np
import matplotlib.pyplot as plt


p = 0.5
q = 1 - p
n_steps = 100
x0 = 0
delta_x = 1

probability = np.zeros(23, dtype=np.float64)
probability[x0 + 11] = 1.0
probability_new = np.zeros(23, dtype=np.float64)

for _ in range(n_steps):
    for i in range(len(probability)):
        if i == 0:
            probability_new[i] = q * probability[i + 1] + probability[i]

        elif i == 1:
            probability_new[i] = q * probability[i + 1]

        elif i == len(probability) - 2:
            probability_new[i] = p * probability[i - 1]

        elif i == len(probability) - 1:
            probability_new[i] = p * probability[i - 1] + probability[i]

        else:
            probability_new[i] = p * probability[i - 1] + q * probability[i + 1]

    probability = probability_new.copy()
    
right_trap_probability = probability[-1]
print(right_trap_probability)
    


positions = np.arange(-11, 12)
probability = np.array(probability, dtype=np.float64).reshape(1, 23)
fig, ax = plt.subplots(figsize=(12, 2))
heatmap = ax.imshow(probability, cmap="viridis", aspect="auto")
plt.colorbar(heatmap, ax=ax, orientation="vertical", pad=0.05)
ax.set_xticks(np.arange(23))
ax.set_xticklabels(positions)
ax.set_xlabel("Position")
ax.set_yticks([])
plt.title(f"Probability Distribution after {n_steps} steps (p={p})")
ax.set_xticks(np.arange(-0.5, 23, 1), minor=True)
ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
plt.tight_layout()
plt.show()
