import pygame
import random
import numpy as np


random.seed(42)
n_steps = 10000
radius = 6
m = 1
v_max = 1
sigma = 1
epsilon = 1
k_B = 1
dt = 0.01

pygame.init()
w, h = 1550, 880
background_color = (11, 10, 34)
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Lennard Jones Simulation")

# Yellow, Blue, Pink, Green
particle_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
n_particles = 100


class Particle:
    def __init__(self, x, y, v0_x, v0_y, fx, fy, radius, m, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = v0_x
        self.vy = v0_y
        self.fx = fx
        self.fy = fy
        self.m = m

    def move(self):
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Periodic boundary conditions
        self.x = self.x % w
        self.y = self.y % h

    def apply_force(self, fx, fy):
        self.vx += fx * dt / self.m
        self.vy += fy * dt / self.m

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


def F(r_vec):
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros_like(r_vec)
    f = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r**2
    f_vec = f * r_vec / r
    return f_vec


def velocity_verlet(x_old, v_old, dt=dt):
    """
    Velocity Verlet algorithm in 1D
    """
    a = F(x_old) / m
    x_new = x_old + v_old * dt + 0.5 * a * dt**2

    a_new = F(x_new) / m
    v_new = v_old + 0.5 * (a + a_new) * dt
    return x_new, v_new


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
while running and i < n_steps:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(background_color)
    for particle in particles:
        particle.draw(screen)

    # Calc forces
    for particle in particles:
        fx = 0
        fy = 0
        for other in particles:
            if other != particle:
                r_vec = distance(particle.x, particle.y, other.x, other.y)
                f_vec = F(r_vec)
                fx += f_vec[0]
                fy += f_vec[1]
        particle.apply_force(fx, fy)

    # Update positions
    for particle in particles:
        particle.x, particle.vx = velocity_verlet(particle.x, particle.vx)
        particle.y, particle.vy = velocity_verlet(particle.y, particle.vy)

    for particle in particles:
        particle.move()

    pygame.display.flip()
    clock.tick(60)
    i += 1

pygame.quit()
