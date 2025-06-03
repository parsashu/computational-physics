import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


random.seed(42)
N = 49
n_steps = 100000
m = 1
v_max = 5
sigma = 30
radius = 11
r_cutoff = 2.5 * sigma
epsilon = 50
k_B = 1
dt = 0.1

pygame.init()
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
    for i in range(N):
        for j in range(i + 1, N):
            d = distance(particles[i].r, particles[j].r)
            d_norm = np.linalg.norm(d)

            if d_norm < r_cutoff:
                potential += (
                    4 * epsilon * ((sigma / d_norm) ** 12 - (sigma / d_norm) ** 6)
                )
    return potential


def Vacf(velocity_list, max_tau=None):
    """
    Calculate velocity autocorrelation function (VACF)
    """
    if max_tau is None:
        max_tau = len(velocity_list) // 2

    vacf = np.zeros(max_tau)
    for tau in tqdm(range(max_tau), desc="Calculating VACF"):
        correlation = 0
        count = 0

        for t in range(len(velocity_list) - tau):
            for p in range(N):
                v0 = velocity_list[t][p]
                vt = velocity_list[t + tau][p]
                correlation += np.dot(v0, vt)
                count += 1

        if count > 0:
            vacf[tau] = correlation / count

    return vacf


def diffusion(vacf, dt):
    """
    Calculate diffusion coefficient by integrating VACF
    """
    D = 0.25 * dt * np.trapz(vacf)
    return D


def Equilibration_time(vacf, threshold=1 / np.e):
    for t, value in enumerate(vacf):
        if abs(value) < threshold:
            return t
    return len(vacf)


def temperature(kinetic):
    """
    Calculate the temperature of the system
    """
    return kinetic / (N * k_B)


def pressure(particles, temp):
    """
    Calculate the pressure of the system
    """
    virial = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = distance(particles[i].r, particles[j].r)
            d_norm = np.linalg.norm(d)
            if d_norm < r_cutoff:
                F = F_ij(d)
                virial += np.dot(F, d)
    V = w * h
    P = (N * k_B * temp + 0.5 * virial) / V
    return P


# Initial velocities
particles = []
v_list = np.random.uniform(-v_max, v_max, (2, N))

# Set Vcm = 0
v_list -= np.mean(v_list, axis=1, keepdims=True)

# Initial positions
left_width = w // 2
min_spacing = 3 * radius

max_cols = left_width // min_spacing
max_rows = h // min_spacing

if max_cols * max_rows >= N:
    cols = min(max_cols, int(np.ceil(np.sqrt(N * left_width / h))))
    rows = int(np.ceil(N / cols))
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
        if i >= N:
            break

        x0 = col * spacing_x + spacing_x // 2
        y0 = row * spacing_y + spacing_y // 2

        x0 = max(radius, min(x0, left_width - radius))
        y0 = max(radius, min(y0, h - radius))

        color = particle_colors[i % len(particle_colors)]
        particles.append(
            Particle(
                np.array([x0, y0], dtype=np.float64),
                v_list[:, i],
                np.zeros(2, dtype=np.float64),
                radius,
                m,
                color,
            )
        )
        i += 1
    if i >= N:
        break

# Initial accelerations
for particle in particles:
    particle.a = F_tot(particle) / particle.m


clock = pygame.time.Clock()
energy_list = []
kinetic_list = []
potential_list = []
n_left_list = []
velocity_list = []
temp_list = []
pres_list = []
running = True

# Main Loop
for i in tqdm(range(n_steps), desc="MD Simulation", unit="steps"):
    if not running:
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    screen.fill(background_color)
    for particle in particles:
        particle.draw(screen)

    # Calculate energy
    # kinetic = kinetic_energy(particles)
    # potential = potential_energy(particles)
    # energy = kinetic + potential
    # energy_list.append(energy)
    # kinetic_list.append(kinetic)
    # potential_list.append(potential)

    # # Number of particles on the right side
    # n_left = n_particles_left_side(particles)
    # n_left_list.append(n_left)

    # VACF
    # current_velocities = []
    # for particle in particles:
    #     current_velocities.append(particle.v)
    # velocity_list.append(current_velocities)

    # Temperature and pressure
    # temp = temperature(kinetic)
    # pres = pressure(particles, temp)
    # temp_list.append(temp)
    # pres_list.append(pres)

    # Update positions
    for particle in particles:
        particle.move()

    pygame.display.flip()
    clock.tick(1200)

pygame.quit()


if len(n_left_list) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(n_left_list)), n_left_list, "b-", label="Particles on Left Side")
    plt.axhline(y=50, color="r", linestyle="--", label="Equilibrium")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Particles")
    plt.title("Number of Particles on Left Side Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if len(energy_list) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(energy_list)), energy_list, "b-", label="Total Energy")
    plt.plot(range(len(kinetic_list)), kinetic_list, "r-", label="Kinetic Energy")
    plt.plot(range(len(potential_list)), potential_list, "g-", label="Potential Energy")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.title("System Energy Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if len(pres_list) > 0:
    equ_time = 1000
    mean_pres = np.mean(pres_list[equ_time:])
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(pres_list)),
        pres_list,
        "r-",
        label=f"Pressure (mean: {mean_pres:.4f})",
    )
    plt.axhline(y=mean_pres, color="g", linestyle="--", label="Mean Pressure")
    plt.xlabel("Time Step")
    plt.ylabel("Pressure")
    plt.title("Pressure over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if len(temp_list) > 0:
    equ_time = 1000
    mean_temp = np.mean(temp_list[equ_time:])
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(temp_list)),
        temp_list,
        "b-",
        label=f"Temperature (mean: {mean_temp:.4f})",
    )
    plt.axhline(y=mean_temp, color="g", linestyle="--", label="Mean Temperature")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.title("Temperature over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if len(velocity_list) > 0:
    vacf = Vacf(velocity_list, 1000)
    equ_time = Equilibration_time(vacf)

    # Calculate diffusion coefficient
    D = diffusion(vacf, dt)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(vacf)), vacf, "b-", label="VACF")
    plt.axvline(
        x=equ_time,
        color="r",
        linestyle="--",
        label=f"Equilibration Time: {equ_time} steps",
    )
    plt.xlabel("Time Step (τ)")
    plt.ylabel("Velocity Autocorrelation")
    plt.title(f"Velocity Autocorrelation Function\nDiffusion Coefficient D = {D:.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()
