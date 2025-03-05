import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit


N = 1000
L = 200

# Create an array of 200 zeros for height tracking
surface = np.zeros(L)
surface[L // 2] = 1

# Create a 3D array to track particle colors (0: no particle, 1: blue, 2: light blue)
particle_colors = np.zeros((1, L), dtype=int)
particle_colors[0, L // 2] = 1

# Create arrays to track width over time
w_array = np.zeros(N)



# Function to add a particle and track its color
def add_particle(position, color_value):
    global particle_colors, particle_count
    height = int(surface[position])

    # Wrap boundaries
    if position == L - 1:  # position = 199
        previous = position - 1
        next = 0
    else:
        previous = position - 1
        next = position + 1

    # Determine the maximum height needed
    max_height_needed = height
    if surface[next] > surface[position]:
        max_height_needed = int(surface[next])
    elif surface[previous] > surface[position]:
        max_height_needed = int(surface[previous])

    # Expand particle_colors array if needed
    while max_height_needed >= particle_colors.shape[0]:
        additional_height = particle_colors.shape[0]  # Double the size
        zeros_to_append = np.zeros((additional_height, L), dtype=int)
        particle_colors = np.vstack((particle_colors, zeros_to_append))

    # Add the particle with its color
    if surface[next] > surface[position]:
        max_height = int(surface[next])
        particle_colors[max_height, position] = color_value
        surface[position] += max_height - height
        # Increment particle count at this position
        particle_count[position] += 1

    elif surface[previous] > surface[position]:
        max_height = int(surface[previous])
        particle_colors[max_height, position] = color_value
        surface[position] += max_height - height
        # Increment particle count at this position

    else:
        particle_colors[height, position] = color_value
        surface[position] += 1
        # Increment particle count at this position


def calculate_width():
    mean_height = np.mean(surface)
    mean_height_squared = np.mean(surface**2)
    w = (mean_height_squared - mean_height**2) ** 0.5
    return w


# Deposit particles with alternating colors
for i in range(N):
    # Choose a random integer between 0 and 199
    random_position = np.random.randint(0, L)

    # Determine color based on deposition time
    if (i // (N / 4)) % 2 == 0:
        color = 1  # Blue
    else:
        color = 2  # Light blue

    # Add the particle
    add_particle(random_position, color)

    w_array[i] = calculate_width()


# # Filter out zero values to avoid log(0) issues--------------------------------
# nonzero_indices = np.where(w_array > 0)[0]
# time_data = nonzero_indices + 1  # +1 to avoid log(0)
# width_data = w_array[nonzero_indices]

# # Convert to log space
# log_time = np.log10(time_data)
# log_width = np.log10(width_data)

# # Plot the raw data in log-log space
# plt.plot(log_time, log_width, "b.", alpha=0.5, label="Data")

# # Fit constant function (using mean of log_width)
# constant_value = np.mean(log_width)

# # Generate fitted line points
# x_fit = np.linspace(log_time[0], log_time[-1], 100)
# y_fit = np.full_like(x_fit, constant_value)

# # Plot the constant fit
# plt.plot(
#     x_fit,
#     y_fit,
#     "r-",
#     label=f"Constant Fit: y = {constant_value:.3f}",
# )

# plt.xlabel("log(Number of Deposited Particles)")
# # Increase number of x-axis ticks
# x_min, x_max = plt.xlim()
# plt.xticks(np.linspace(x_min, x_max, 15))  # 15 ticks from min to max

# plt.ylabel("log(Surface Width)")

# # Increase number of y-axis ticks
# y_min, y_max = plt.ylim()
# plt.yticks(np.linspace(y_min, y_max, 15))  # 15 ticks from min to max

# plt.title(f"Log-Log Plot of Surface Width vs Number of Particles (L={L}, N={N})")

# plt.grid(True)
# plt.legend()
# plt.show()


# # Plot the width evolution over time to find, beta N = 2000----------------------
# plt.figure(figsize=(10, 6))
# plt.plot(range(N), w_array, "r-", alpha=0.7)
# plt.xlabel("Number of Deposited Particles")
# plt.ylabel("Surface Width (w)")
# plt.title(f"Surface Width Evolution in Lateral Deposition (Total Particles: {N})")
# plt.grid(True)


# # Define the power law function: w(t) = A * t^beta
# def power_law(t, A, beta):
#     return A * t**beta


# # Use only the latter part of the data for fitting (after initial transient)
# fit_start = N // 10  # Start fitting from 10% of the data
# x_data = np.arange(fit_start, N)
# y_data = w_array[fit_start:]

# # Perform the curve fitting
# params, covariance = curve_fit(power_law, x_data, y_data)
# A_fit, beta_fit = params
# beta_error = np.sqrt(np.diag(covariance))[1]  # Extract the error in beta

# # Generate the fitted curve
# y_fit = power_law(x_data, A_fit, beta_fit)

# # Plot the fitted curve
# plt.plot(x_data, y_fit, "b--", label=f"Fitted: t^{beta_fit:.3f}±{beta_error:.3f}")
# plt.legend()

# print(f"Fitted growth exponent (beta): {beta_fit:.4f} ± {beta_error:.4f}")
# print(f"Amplitude (A): {A_fit:.6e}")
# plt.show()

# Create a visualization-------------------------------------------------
max_height = int(np.max(surface))

# Create a custom colormap: 0=white, 1=blue, 2=light blue
colors = ["white", "blue", "skyblue"]
cmap = ListedColormap(colors)

# Plot the pixel-based surface
plt.figure(figsize=(10, 6))

# Trim the particle_colors array to only include the heights we need
particle_colors_trimmed = particle_colors[:max_height, :]

plt.imshow(particle_colors_trimmed, cmap=cmap, interpolation="none", origin="lower")
plt.xlabel("Position")
plt.ylabel("Height")

# Set y-ticks to show actual heights (no need for custom calculation now)
num_ticks = 5  # Adjust this for more or fewer ticks
plt.yticks(np.linspace(0, max_height - 1, num_ticks).astype(int))

plt.title(f"Lateral Deposition Model (Total Particles: {N})")
plt.show()
