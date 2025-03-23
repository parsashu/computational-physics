import numpy as np
import matplotlib.pyplot as plt

p = 0.5
l = 1
T = 1
q = 1 - p
x0 = -10
delta_x = 1
trap_x = 10
ensamble = 10000


def single_random_walk(x0):
    """Simulates a single random walk."""
    position = x0
    path = [x0]

    while position <= trap_x and position >= -trap_x:
        step_direction = np.random.choice([-l, l], p=[q, p])
        position += step_direction
        path.append(position)

    times = [step * T for step in range(len(path))]
    return path, times


def mean_life_time(ensamble, x0):
    """Calculates the mean life time over multiple trials."""
    life_times = []

    for _ in range(ensamble):
        _, times = single_random_walk(x0)
        life_time = times[-1]
        life_times.append(life_time)

    mean_life_time = np.mean(life_times)
    return mean_life_time


def theorical_mean(times):
    return (l / T) * (p - q) * np.array(times)


def theoretical_variance(times):
    return 4 * l**2 * (p * q) * (1 / T) * np.array(times)


mean_life_times_list = []
x0_list = []

while x0 <= trap_x:
    mean_life_time_ = mean_life_time(ensamble, x0)

    mean_life_times_list.append(mean_life_time_)
    x0_list.append(x0)

    x0 += delta_x

plt.figure(figsize=(10, 6))
plt.plot(x0_list, mean_life_times_list, "o")
plt.title(
    f"Mean Life Time vs Initial Position\np={p}"
)
plt.xlabel("Initial Position (x₀)")
plt.ylabel("Mean Life Time")
plt.grid(True)
plt.show()
