import pygame
import sys
import numpy as np
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800.0, 600.0
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 100, 255)
YELLOW = (255, 255, 150)

# Car Properties
axle_sep = 60.0
wheel_sep = 0.6*axle_sep

wheel_dep = wheel_sep/4  # how fat will the wheels be
wheel_diam = wheel_dep*3.0  # how big are they in diameter


rr_anchor = (0.0 - wheel_diam/2., 0.0 - wheel_sep/2. - wheel_dep/2.)
lr_anchor = (0.0 - wheel_diam/2., 0.0 + wheel_sep/2. - wheel_dep/2.)
rf_anchor = (0.0 + axle_sep - wheel_diam/2., 0.0 - wheel_sep/2. - wheel_dep/2.)
lf_anchor = (0.0 + axle_sep - wheel_diam/2., 0.0 + wheel_sep/2. - wheel_dep/2.)

base_LR_pad = wheel_dep/2.0
base_UD_pad = -wheel_dep/4.0
caranchor = (0.0 - wheel_diam/2.0 - base_LR_pad, 0.0 - wheel_sep/2.0 - wheel_dep/2.0 - base_UD_pad)
car_len = axle_sep + wheel_diam + base_LR_pad*2.0
car_width = wheel_sep + base_UD_pad*2.0 + wheel_dep
car_rr = np.array(caranchor)
car_lr = car_rr + np.array([0.0, car_width])
car_rf = car_rr + np.array([car_len, 0.15*car_width])
car_lf = car_lr + np.array([car_len, -0.15*car_width])
car_pts = np.vstack([car_rr, car_lr, car_lf, car_rf])

def blit_rotate_center(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)

# Sprite class
class CarSprite(pygame.sprite.Sprite):
    def __init__(self, center):
        super().__init__()
        self.state = np.zeros(5)
        self.center = center
        self.image = self.create_image()
        self.rect = self.image.get_rect(center=self.center)
        self.state[0] = self.rect.x + self.image.get_rect().width/2
        self.state[1] = self.rect.y + self.image.get_rect().height/2

    def create_image(self):
        max_width = 2*car_width
        max_height = 2*car_len
        image = pygame.Surface((max_width, max_height), pygame.SRCALPHA)
        pygame.draw.rect(image, BLUE, (car_width/2, car_len/2, car_width, car_len))

        wheel_fl = pygame.Surface((wheel_dep, wheel_diam), pygame.SRCALPHA)
        wheel_fl.fill(BLACK)
        blit_rotate_center(image, wheel_fl, (car_width - wheel_sep/2. - wheel_dep/2., car_len - axle_sep/2 - wheel_diam/2), self.state[4])

        wheel_fr = pygame.Surface((wheel_dep, wheel_diam), pygame.SRCALPHA)
        wheel_fr.fill(BLACK)
        blit_rotate_center(image, wheel_fr, (car_width + wheel_sep/2. - wheel_dep/2., car_len - axle_sep/2 - wheel_diam/2), self.state[4])

        pygame.draw.rect(image, BLACK, (car_width - wheel_sep/2. - wheel_dep/2., car_len + axle_sep/2 - wheel_diam/2, wheel_dep, wheel_diam))
        pygame.draw.rect(image, BLACK, (car_width + wheel_sep/2. - wheel_dep/2., car_len + axle_sep/2 - wheel_diam/2, wheel_dep, wheel_diam))
        return image

    def update(self, keys):
        if keys[pygame.K_UP]:
            self.state[3] += 0.1
        if keys[pygame.K_DOWN]:
            self.state[3] -= 0.1
        if keys[pygame.K_LEFT]:
            self.state[4] += 0.5
        if keys[pygame.K_RIGHT]:
            self.state[4] -= 0.5

        self.image = pygame.transform.rotate(self.create_image(), self.state[2])
        self.rect = self.image.get_rect(center=self.rect.center)

        self.state[0] -= self.state[3] * math.sin(math.radians(self.state[2]))
        self.state[1] -= self.state[3] * math.cos(math.radians(self.state[2]))
        self.state[2] += 100* self.state[3] * math.tan(math.radians(self.state[4])) / axle_sep

        self.rect.x = self.state[0] - self.image.get_rect().width/2
        self.rect.y = self.state[1] - self.image.get_rect().height/2

        # print(self.rect.y, self.rect.x, self.state[3] * math.cos(math.radians(self.state[2])), self.state[3] * math.sin(math.radians(self.state[2])))

# Clock to control the frame rate
clock = pygame.time.Clock()

center_point = (WIDTH // 2, HEIGHT // 2)
sprite = CarSprite(center_point)
all_sprites = pygame.sprite.Group(sprite)

# Clock to control the frame rate
clock = pygame.time.Clock()
FPS = 60

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    all_sprites.update(keys)

    screen.fill(WHITE)
    all_sprites.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()