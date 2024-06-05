import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sprite with Multiple Rectangles Rotating Around a Center Point")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Sprite class
class MultiRectSprite(pygame.sprite.Sprite):
    def __init__(self, rects, color, center):
        super().__init__()
        self.rects = rects
        self.color = color
        self.angle = 0
        self.center = center
        self.image = self.create_image()
        self.rect = self.image.get_rect(center=self.center)

    def create_image(self):
        max_width = max([rect[0] + rect[2] for rect in self.rects])
        max_height = max([rect[1] + rect[3] for rect in self.rects])
        max_size = max(max_width, max_height) * 2
        image = pygame.Surface((max_size, max_size), pygame.SRCALPHA)
        for rect in self.rects:
            pygame.draw.rect(image, self.color, rect)
        return image

    def update(self, keys):
        if keys[pygame.K_UP]:
            self.rect.x += 5 * math.cos(math.radians(self.angle))
            self.rect.y -= 5 * math.sin(math.radians(self.angle))
        if keys[pygame.K_DOWN]:
            self.rect.x -= 5 * math.cos(math.radians(self.angle))
            self.rect.y += 5 * math.sin(math.radians(self.angle))
        if keys[pygame.K_LEFT]:
            self.angle += 5
        if keys[pygame.K_RIGHT]:
            self.angle -= 5

        self.image = pygame.transform.rotate(self.create_image(), self.angle)
        self.rect = self.image.get_rect(center=self.center)

# Create the sprite
rects = [(0, 0, 60, 30), (60, 30, 30, 30)]  # Relative positions and sizes of rectangles
center_point = (WIDTH // 2, HEIGHT // 2)
sprite = MultiRectSprite(rects, BLACK, center_point)
all_sprites = pygame.sprite.Group(sprite)

# Clock to control the frame rate
clock = pygame.time.Clock()

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
    clock.tick(60)

pygame.quit()
sys.exit()
