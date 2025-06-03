import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


random.seed(42)
n_particles = 100
n_steps = 10000
m = 1
v_max = 50
sigma = 50
radius = 11
r_cutoff = 10 * sigma
epsilon = 20
k_B = 1
dt = 0.01

pygame.init()
# w, h = 1550, 880
w, h = 1200, 700
background_color = (11, 10, 34)
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Lennard Jones Simulation")

# Yellow, Blue, Pink, Green
particle_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]


class Particle:
    def __init__(self, r, v, a, radius, m, color):
        self.r = r
        self.v = v
        self.a = a
        self.radius = radius
        self.color = color
        self.m = m

    def move(self):
        """
        Velocity Verlet algorithm in 2D
        """
        self.r += self.v * dt + 0.5 * self.a * dt**2

        a_new = F_tot(self) / self.m
        self.v += 0.5 * (self.a + a_new) * dt

        self.a = a_new

        # Periodic boundary conditions
        self.r[0] = self.r[0] % w
        self.r[1] = self.r[1] % h

    def draw(self, screen):
        pygame.draw.circle(
            screen, self.color, (int(self.r[0]), int(self.r[1])), self.radius
        )


def distance(r1, r2):
    """
    Calculate distance with periodic boundary conditions
    """
    dx = r2[0] - r1[0]
    dy = r2[1] - r1[1]

    if dx > w / 2:
        dx -= w
    elif dx < -w / 2:
        dx += w

    if dy > h / 2:
        dy -= h
    elif dy < -h / 2:
        dy += h
    return np.array([dx, dy])


# def distance_matrix(particles):
#     distance_matrix = np.zeros((n_particles, n_particles, 2))
#     for i in range(n_particles):
#         for j in range(i + 1, n_particles):
#             distance_matrix[i, j] = distance(
#                 particles[i].x, particles[i].y, particles[j].x, particles[j].y
#             )
#             distance_matrix[j, i] = -distance_matrix[i, j]
#     return distance_matrix


def F_ij(d):
    d_norm = np.linalg.norm(d)
    if d_norm > r_cutoff:
        return np.zeros(2)
    f = 24 * epsilon * (2 * (sigma / d_norm) ** 12 - (sigma / d_norm) ** 6) / d_norm**2
    f_vec = f * d
    return f_vec


def F_tot(particle):
    f_tot = np.zeros(2)
    for other in particles:
        if other != particle:
            d = distance(other.r, particle.r)
            f = F_ij(d)
            f_tot += f
    return f_tot


# def Force_matrix(particles):
#     force_matrix = np.zeros((n_particles, n_particles, 2))
#     dist_matrix = distance_matrix(particles)
#     for i in range(n_particles):
#         for j in range(i + 1, n_particles):
#             r_vec = dist_matrix[i, j]
#             force_matrix[i, j] = F_ij(r_vec)
#             force_matrix[j, i] = -force_matrix[i, j]
#     return force_matrix


# def F_tot(force_matrix, particle):
#     particle_index = particles.index(particle)
#     fx = np.sum(force_matrix[particle_index, :, 0])
#     fy = np.sum(force_matrix[particle_index, :, 1])
#     return np.array([fx, fy])


def n_particles_left_side(particles):
    n = 0
    for particle in particles:
        if particle.r[0] < w / 2:
            n += 1
    return n


def kinetic_energy(particles):
    """
    Calculate the total kinetic energy of the system
    """
    kinetic = 0
    for particle in particles:
        kinetic += 0.5 * particle.m * np.dot(particle.v, particle.v)
    return kinetic


def potential_energy(particles):
    """
    Calculate the total potential energy of the system
    """
    potential = 0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            d = distance(particles[i].r, particles[j].r)
            d_norm = np.linalg.norm(d)

            if d_norm < r_cutoff:
                potential += (
                    4 * epsilon * ((sigma / d_norm) ** 12 - (sigma / d_norm) ** 6)
                )
    return potential


# Initial velocities
particles = []
v0x_list = np.random.uniform(-v_max, v_max, n_particles)
v0y_list = np.random.uniform(-v_max, v_max, n_particles)

# Set Vcm = 0
v0x_list -= np.mean(v0x_list)
v0y_list -= np.mean(v0y_list)

# Initial positions
left_width = w // 2
min_spacing = 3 * radius

max_cols = left_width // min_spacing
max_rows = h // min_spacing

if max_cols * max_rows >= n_particles:
    cols = min(max_cols, int(np.ceil(np.sqrt(n_particles * left_width / h))))
    rows = int(np.ceil(n_particles / cols))
else:
    cols = max_cols
    rows = max_rows

spacing_x = left_width // cols if cols > 0 else min_spacing
spacing_y = h // rows if rows > 0 else min_spacing

spacing_x = max(spacing_x, min_spacing)
spacing_y = max(spacing_y, min_spacing)

i = 0
for row in range(rows):
    for col in range(cols):
        if i >= n_particles:
            break

        x0 = col * spacing_x + spacing_x // 2
        y0 = row * spacing_y + spacing_y // 2

        x0 = max(radius, min(x0, left_width - radius))
        y0 = max(radius, min(y0, h - radius))

        v0x = v0x_list[i]
        v0y = v0y_list[i]

        color = particle_colors[i % len(particle_colors)]
        particles.append(Particle(x0, y0, v0x, v0y, 0, 0, radius, m, color))
        i += 1

    if i >= n_particles:
        break


clock = pygame.time.Clock()
energy_list = []
kinetic_list = []
potential_list = []
n_left_list = []
velocity_list = []

# Main Loop
for i in tqdm(range(n_steps), desc="MD Simulation", unit="steps"):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break

    screen.fill(background_color)
    for particle in particles:
        particle.draw(screen)

    # Store current velocities
    current_velocities = []
    for particle in particles:
        current_velocities.append(np.array([particle.vx, particle.vy]))
    velocity_list.append(current_velocities)

    # Calculate energy
    kinetic = kinetic_energy(particles)
    potential = potential_energy(particles)
    energy = kinetic + potential
    energy_list.append(energy)
    kinetic_list.append(kinetic)
    potential_list.append(potential)

    # Number of particles on the right side
    # n_left = n_particles_left_side(particles)
    # n_left_list.append(n_left)

    # Update positions
    # force_matrix = Force_matrix(particles)
    for particle in particles:
        particle.move()

    pygame.display.flip()
    clock.tick(1200)

pygame.quit()


if len(n_left_list) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(range(i), n_left_list, "b-", label="Particles on Left Side")
    plt.axhline(y=50, color="r", linestyle="--", label="Equilibrium")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Particles")
    plt.title("Number of Particles on Left Side Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if len(energy_list) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(range(i), energy_list, "b-", label="Total Energy")
    plt.plot(range(i), kinetic_list, "r-", label="Kinetic Energy")
    plt.plot(range(i), potential_list, "g-", label="Potential Energy")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.title("System Energy Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def Vacf(velocity_list, max_tau=None):
    if max_tau is None:
        max_tau = len(velocity_list) // 2

    vacf = np.zeros(max_tau)
    n_particles = len(velocity_list[0])

    for tau in range(max_tau):
        correlation = 0
        count = 0

        for t in range(len(velocity_list) - tau):
            for p in range(n_particles):
                v0 = velocity_list[t][p]
                vt = velocity_list[t + tau][p]
                correlation += np.dot(v0, vt)
                count += 1

        if count > 0:
            vacf[tau] = correlation / count

    return vacf


def Equilibration_time(vacf, threshold=1 / np.e):
    for t, value in enumerate(vacf):
        if abs(value) < threshold:
            return t
    return len(vacf)


# vacf = Vacf(velocity_list)
# equilibration_time = Equilibration_time(vacf)
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(vacf)), vacf, "b-", label="VACF")
# plt.axvline(
#     x=equilibration_time,
#     color="r",
#     linestyle="--",
#     label=f"Equilibration Time: {equilibration_time} steps",
# )
# plt.xlabel("Time Step (τ)")
# plt.ylabel("Velocity Autocorrelation")
# plt.title("Velocity Autocorrelation Function")
# plt.legend()
# plt.grid(True)
# plt.show()
