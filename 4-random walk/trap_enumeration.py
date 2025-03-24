import numpy as np
import matplotlib.pyplot as plt


p = 0.5
q = 1 - p
n_steps = 100
x0 = 0
delta_x = 1


def random_walk_until_threshold(death_prob_threshold, x0, p, q):
    probability = np.zeros(23, dtype=np.float64)
    probability[x0 + 11] = 1.0
    probability_new = np.zeros(23, dtype=np.float64)

    n_steps = 0
    right_trap_prob = 0
    left_trap_prob = 0

    while right_trap_prob + left_trap_prob < death_prob_threshold:
        n_steps += 1

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
        right_trap_prob = probability[-1]
        left_trap_prob = probability[0]

    return n_steps, probability, right_trap_prob, left_trap_prob


death_prob_thresholds = np.linspace(0.1, 0.9, 9)
steps_required = []

for death_prob_threshold in death_prob_thresholds:
    n_steps, _, _, _ = random_walk_until_threshold(death_prob_threshold, x0, p, q)
    steps_required.append(n_steps)

plt.figure(figsize=(10, 6))
plt.plot(death_prob_thresholds, steps_required, "o-", linewidth=2)
plt.xlabel("Death Probability Threshold")
plt.ylabel("Number of Steps Required")
plt.title(f"Steps Required to Reach Different Death Probability Thresholds (p={p})")
plt.grid(True)
plt.show()

death_prob_threshold = 0.68
n_steps, probability, right_trap_prob, left_trap_prob = random_walk_until_threshold(
    death_prob_threshold, x0, p, q
)

positions = np.arange(-11, 12)
probability = np.array(probability, dtype=np.float64).reshape(1, 23)
fig, ax = plt.subplots(figsize=(12, 2))
heatmap = ax.imshow(probability, cmap="viridis", aspect="auto")
plt.colorbar(heatmap, ax=ax, orientation="vertical", pad=0.05)
ax.set_xticks(np.arange(23))
ax.set_xticklabels(positions)
ax.set_xlabel("Position")
ax.set_yticks([])
plt.title(
    f"Probability Distribution after {n_steps} steps (p={p}, threshold={death_prob_threshold:.2f})"
)
ax.set_xticks(np.arange(-0.5, 23, 1), minor=True)
ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
plt.tight_layout()
plt.show()
