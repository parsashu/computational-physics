import pygame
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(42)
n_particles = 100
n_steps = 10000
m = 1
v_max = 10
sigma = 40
radius = 10
r_cutoff = 10 * sigma
epsilon = 10
k_B = 1
dt = 0.01

pygame.init()
# w, h = 1550, 880
w, h = 1000, 700
background_color = (11, 10, 34)
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Lennard Jones Simulation")

# Yellow, Blue, Pink, Green
particle_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]


class Particle:
    def __init__(self, x, y, v0_x, v0_y, ax, ay, radius, m, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = v0_x
        self.vy = v0_y
        self.ax = ax
        self.ay = ay
        self.m = m

    def move(self, force_matrix, particles):
        """
        Velocity Verlet algorithm in 2D
        """
        self.x += self.vx * dt + 0.5 * self.ax * dt**2
        self.y += self.vy * dt + 0.5 * self.ay * dt**2

        ax_new, ay_new = F_tot(force_matrix, particles, self) / self.m
        self.vx += 0.5 * (self.ax + ax_new) * dt
        self.vy += 0.5 * (self.ay + ay_new) * dt

        self.ax, self.ay = ax_new, ay_new

        # Periodic boundary conditions
        self.x = self.x % w
        self.y = self.y % h

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


def distance(x1, y1, x2, y2):
    """
    Calculate distance with periodic boundary conditions
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx > w / 2:
        dx -= w
    elif dx < -w / 2:
        dx += w

    if dy > h / 2:
        dy -= h
    elif dy < -h / 2:
        dy += h
    return np.array([dx, dy])


def F_ij(r_vec):
    r = np.linalg.norm(r_vec)
    if r > r_cutoff:
        return np.zeros(2)
    f = -24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
    f_vec = f * r_vec / r
    return f_vec


def Force_matrix(particles):
    force_matrix = np.zeros((len(particles), len(particles), 2))
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            r_vec = distance(
                particles[i].x, particles[i].y, particles[j].x, particles[j].y
            )
            force_matrix[i, j] = F_ij(r_vec)
            force_matrix[j, i] = -force_matrix[i, j]
    return force_matrix


def F_tot(force_matrix, particles, particle):
    particle_index = particles.index(particle)
    fx = np.sum(force_matrix[particle_index, :, 0])
    fy = np.sum(force_matrix[particle_index, :, 1])
    return np.array([fx, fy])


def n_particles_left_side(particles):
    n = 0
    for particle in particles:
        if particle.x < w / 2:
            n += 1
    return n


def kinetic_energy(particles):
    """
    Calculate the total kinetic energy of the system
    """
    kinetic = 0
    for particle in particles:
        v_squared = particle.vx**2 + particle.vy**2
        kinetic += 0.5 * particle.m * v_squared
    return kinetic


def potential_energy(particles):
    """
    Calculate the total potential energy of the system
    """
    potential = 0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            r_vec = distance(
                particles[i].x, particles[i].y, particles[j].x, particles[j].y
            )
            r = np.linalg.norm(r_vec)

            if r < r_cutoff:
                potential += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    return potential


# Init particles
particles = []
v0x_list = np.zeros(n_particles)
v0y_list = np.zeros(n_particles)

for i in range(n_particles):
    v0x_list[i] = random.uniform(-v_max, v_max)
    v0y_list[i] = random.uniform(-v_max, v_max)

# Set Vcm = 0
mean_v0x = np.mean(v0x_list)
mean_v0y = np.mean(v0y_list)
v0x_list = v0x_list - mean_v0x
v0y_list = v0y_list - mean_v0y

i = 0
rows = int(np.sqrt(2 * n_particles * h / w))
cols = int(np.sqrt(2 * n_particles * w / h))
spacing = int(np.sqrt((w * h) / (2 * n_particles)))

for row in range(rows):
    for col in range(cols // 2):
        x0 = col * spacing + spacing // 2
        y0 = row * spacing + spacing // 2
        v0x = v0x_list[i]
        v0y = v0y_list[i]
        i += 1

        color = particle_colors[i % len(particle_colors)]
        particles.append(Particle(x0, y0, v0x, v0y, 0, 0, radius, m, color))


clock = pygame.time.Clock()
running = True

# Main Loop
i = 0
energy_list = []
kinetic_list = []
potential_list = []
n_left_list = []
velocity_list = []

while running and i < n_steps:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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
    n_left = n_particles_left_side(particles)
    n_left_list.append(n_left)

    # Update positions
    force_matrix = Force_matrix(particles)
    for particle in particles:
        particle.move(force_matrix, particles)

    pygame.display.flip()
    clock.tick(60)
    i += 1

pygame.quit()


plt.figure(figsize=(10, 6))
plt.plot(range(i), n_left_list, "b-", label="Particles on Left Side")
plt.axhline(y=50, color="r", linestyle="--", label="Equilibrium")
plt.xlabel("Time Step")
plt.ylabel("Number of Particles")
plt.title("Number of Particles on Left Side Over Time")
plt.legend()
plt.grid(True)
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(range(i), energy_list, "b-", label="Total Energy")
# plt.plot(range(i), kinetic_list, "r-", label="Kinetic Energy")
# plt.plot(range(i), potential_list, "g-", label="Potential Energy")
# plt.xlabel("Time Step")
# plt.ylabel("Energy")
# plt.title("System Energy Over Time")
# plt.legend()
# plt.grid(True)
# plt.show()


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
