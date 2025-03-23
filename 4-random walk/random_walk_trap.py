import numpy as np
import matplotlib.pyplot as plt

p = 0.5
l = 1
T = 1
q = 1 - p
x0 = -10
delta_x = 1
trap_x = 10
ensamble = 1000


def random_walk(x0):
    """Simulates a single random walk."""
    position = x0
    path = [x0]
    trap_id = 0

    while position <= trap_x and position >= -trap_x:
        step_direction = np.random.choice([-l, l], p=[q, p])
        position += step_direction
        path.append(position)

    if path[-1] > trap_x:
        trap_id = 1
    elif path[-1] < -trap_x:
        trap_id = -1
    times = [step * T for step in range(len(path))]
    return path, times, trap_id


def life_time_and_trap_prob(ensamble, x0):
    """Calculates the mean life and trap probability over multiple trials."""
    life_times = []
    n_right_trap = 0

    for _ in range(ensamble):
        _, times, trap_id = random_walk(x0)
        life_time = times[-1]
        life_times.append(life_time)

        if trap_id == 1:
            n_right_trap += 1

    mean_life_time = np.mean(life_times)
    right_trap_probability = n_right_trap / ensamble
    return mean_life_time, right_trap_probability


mean_life_times_list = []
x0_list = []
trap_ids_list = []
right_trap_probabilities_list = []

while x0 <= trap_x:
    mean_life_time_, right_trap_probability_ = life_time_and_trap_prob(ensamble, x0)

    mean_life_times_list.append(mean_life_time_)
    right_trap_probabilities_list.append(right_trap_probability_)
    x0_list.append(x0)

    x0 += delta_x

left_trap_probabilities_list = 1 - np.array(right_trap_probabilities_list)

plt.figure(figsize=(10, 6))
plt.plot(x0_list, right_trap_probabilities_list, "o", label="Right trap")
plt.plot(x0_list, left_trap_probabilities_list, "o", label="Left trap")
plt.title(f"Trap Probabilities vs Initial Position\np={p}")
plt.xlabel("Initial Position (x₀)")
plt.ylabel("Trap Probability")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x0_list, mean_life_times_list, "o")
plt.title(f"Mean Life Time vs Initial Position\np={p}")
plt.xlabel("Initial Position (x₀)")
plt.ylabel("Mean Life Time")
plt.grid(True)
plt.show()
