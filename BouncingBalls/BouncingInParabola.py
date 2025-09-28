import pygame
from pygame.math import Vector2
import random

pygame.init()

# Set up the drawing window
width, height = 500, 500
screen = pygame.display.set_mode([width, height])

pygame.display.set_caption("Bouncing Ball")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0,0,0)

class Ball(pygame.sprite.Sprite):
    def __init__(self, pos:list, velocity:list, radius=10, color=WHITE, masscharge=1):
        super().__init__()

        self.pos = Vector2(pos[0],pos[1])
        self.vel = Vector2(velocity[0],velocity[1])

        self.radius = radius
        self.color = color
        self.masscharge = masscharge

    def display(self):
        pygame.draw.circle(surface = screen, 
                           color = self.color, 
                           center = self.pos,
                           radius = self.radius)

    def check_for_boundary_collision(self, width, height):

        BF = 0.95 # Bounce Factor, like a coefficient of restitution

        if self.pos[0] + self.radius >= width: # Collision with Right wall
            right_wall_normal = Vector2(-1,0)
            self.pos[0] = 2 * (width - self.radius) - self.pos[0]

            self.vel -= 2 * Vector2.dot(right_wall_normal, self.vel) * right_wall_normal * BF

        elif self.pos[0] - self.radius <= 0: # collision with Left Wall
            left_wall_normal = Vector2(1,0)

            self.vel -= 2 * Vector2.dot(left_wall_normal, self.vel) * left_wall_normal * BF
            self.pos[0] = 2 * self.radius - self.pos[0]

        elif self.pos[1] + self.radius >= height: # Collision with Ground
            ground_normal = Vector2(0,1)
            self.pos[1] = 2 * (height - self.radius) - self.pos[1]

            self.vel -= 2 * Vector2.dot(ground_normal, self.vel) * ground_normal * BF

        elif self.pos[1] - self.radius <= 0: # Collision with Ceiling
            ceiling_normal = Vector2(0,-1)
            self.pos[1] = 2 * self.radius - self.pos[1]
            self.vel -= 2 * Vector2.dot(ceiling_normal, self.vel) * ceiling_normal * BF



# Ball properties
ball_pos = [width/4, height/2]

proton_pos = [width/2,height/2]

ball = Ball(pos=ball_pos, velocity=[4,4])
proton = Ball(pos=proton_pos, velocity=[0,0], color=BLUE)


direction_tick = 0.0
# Run until the user asks to quit
running = True
while running:

    # Limit frames per second
    dtime_ms = pygame.time.Clock().tick(30)
    dtime = dtime_ms

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            print("Click!")

    # Update ball position
    screen.fill(BLACK)
    
    ball.display()
    proton.display()

    #ball.check_for_boundary_collision(width, height)

    force = -(4000/(Vector2.magnitude(ball.pos-proton.pos))**2)*Vector2.normalize(ball.pos-proton.pos)

    ball.pos += ball.vel
    ball.vel += force


    direction_tick += dtime 
    if direction_tick > 1.0:
        direction_tick = 0.0

    pygame.display.flip()

    

# Done! Time to quit.
pygame.quit()