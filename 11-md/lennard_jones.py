import pygame
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(42)
n_particles = 100
n_steps = 1000
radius = 6
m = 1
v_max = 1
sigma = 40
r_cutoff = 10 * sigma
epsilon = 10
k_B = 1
dt = 1

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

    def move(self):
        """
        Velocity Verlet algorithm in 2D
        """
        self.x += self.vx * dt + 0.5 * self.ax * dt**2
        self.y += self.vy * dt + 0.5 * self.ay * dt**2

        ax_new, ay_new = F_tot(self) / self.m
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
    f = -24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r**2
    f_vec = f * r_vec / r
    return f_vec


def F_tot(particle):
    fx = 0
    fy = 0
    for other in particles:
        if other != particle:
            r_vec = distance(particle.x, particle.y, other.x, other.y)
            f_vec = F_ij(r_vec)
            fx += f_vec[0]
            fy += f_vec[1]
    return np.array([fx, fy])


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
spacing = int(np.sqrt((w * h) / n_particles))
rows = int(np.sqrt(n_particles * h / w))
cols = int(n_particles / rows)

for i in range(n_particles):
    v0x_list[i] = random.uniform(-v_max, v_max)
    v0y_list[i] = random.uniform(-v_max, v_max)

# Set Vcm = 0
mean_v0x = np.mean(v0x_list)
mean_v0y = np.mean(v0y_list)
v0x_list = v0x_list - mean_v0x
v0y_list = v0y_list - mean_v0y

i = 0
for row in range(rows):
    for col in range(cols // 2):
        x0 = col * spacing + spacing // 2
        y0 = row * spacing + spacing // 2
        v0x = v0x_list[i]
        v0y = v0y_list[i]
        i += 1

        color = particle_colors[(row + col) % len(particle_colors)]
        particles.append(Particle(x0, y0, v0x, v0y, 0, 0, radius, m, color))


clock = pygame.time.Clock()
running = True

# Main Loop
i = 0
energy_list = []
kinetic_list = []
potential_list = []

while running and i < n_steps:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(background_color)
    for particle in particles:
        particle.draw(screen)

    # Calculate energy
    kinetic = kinetic_energy(particles)
    potential = potential_energy(particles)
    energy = kinetic + potential
    energy_list.append(energy)
    kinetic_list.append(kinetic)
    potential_list.append(potential)

    # Update positions
    for particle in particles:
        particle.move()

    pygame.display.flip()
    clock.tick(60)
    i += 1

pygame.quit()


# Calculate and plot energy
plt.figure(figsize=(10, 6))
plt.plot(range(n_steps), energy_list, "b-", label="Total Energy")
plt.plot(range(n_steps), kinetic_list, "r-", label="Kinetic Energy")
plt.plot(range(n_steps), potential_list, "g-", label="Potential Energy")
plt.xlabel("Time Step")
plt.ylabel("Energy")
plt.title("System Energy Over Time")
plt.legend()
plt.grid(True)
plt.show()
