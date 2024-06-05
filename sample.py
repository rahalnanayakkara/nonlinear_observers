import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rectangle Animation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Rectangle properties
rect_width = 60
rect_height = 30
rect_x = WIDTH // 2
rect_y = HEIGHT // 2
rect_speed = 5
rect_angle = 0  # Initial angle

# Clock to control the frame rate
clock = pygame.time.Clock()

def blit_rotate_center(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Key press handling
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        rect_x += rect_speed * math.cos(math.radians(rect_angle))
        rect_y -= rect_speed * math.sin(math.radians(rect_angle))
    if keys[pygame.K_DOWN]:
        rect_x -= rect_speed * math.cos(math.radians(rect_angle))
        rect_y += rect_speed * math.sin(math.radians(rect_angle))
    if keys[pygame.K_LEFT]:
        rect_angle += 5
    if keys[pygame.K_RIGHT]:
        rect_angle -= 5

    # Fill the screen with white
    screen.fill(WHITE)

    # Create the rectangle surface
    rect_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
    rect_surface.fill(BLACK)

    # Rotate and draw the rectangle
    blit_rotate_center(screen, rect_surface, (rect_x - rect_width // 2, rect_y - rect_height // 2), rect_angle)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
