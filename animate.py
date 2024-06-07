import pygame
import sys
import numpy as np
import math
from numpy.polynomial import Polynomial as P
from nonlinear_system.sample_odes import AckermanModel
from moving_polyfit.moving_ls import MultiDimPolyEstimator

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1600.0, 1200.0
FPS = 30
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animation")

# Estimator Parameters
window_length = 10  # Number of samples
delay = 1
sampling_dt = 1.0/FPS
eval_time = (window_length-1-delay)*sampling_dt
window_times = np.linspace(0., window_length*sampling_dt, window_length, endpoint=False)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)
BLUE = (100, 100, 255)
YELLOW = (255, 255, 150)
RED = (255, 0, 0)
GREEN = (0, 150, 0)

ARROW_LEN = 100

# Font Settings
font_size = 25
font = pygame.font.Font(None, font_size)

# Car Properties
axle_sep = 60.0
noise_mag = 1

steering_limit = 45  # Maximum steering angle in degrees
speed_limit = 7  # Maximum speed of car allowed
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

ODE = AckermanModel(axle_sep, wheel_sep)
ode_n = ODE.n  # system state dimension
ode_m = ODE.m  # control input dimension
ode_p = ODE.p  # output dimension
est_d = 3
poly_estimator = MultiDimPolyEstimator(ode_p, est_d, window_length, sampling_dt)

num_t_points = est_d + 1
deltas = window_length//num_t_points
l_bound = np.zeros((window_length, est_d, deltas))
verbose_lagrange = False  # to see computation details of lagrange polynomial construction/derivatives

for delta in range(1, deltas+1):
    # for index slicing into the time arrays
    maxstart = window_length-1-num_t_points*delta
    minstart = 0
    start = np.clip((window_length-1) - delay - delta*(num_t_points//2), minstart, maxstart)
    l_indices = np.full((num_t_points,), 1)
    for i in range(num_t_points):
        l_indices[i] = start + i*delta
    l_times = window_times[l_indices]  # pull the subset of chosen time indices

    for i in range(num_t_points):
        # build the lagrange polynomial, which is zero at all evaluation samples except one
        evals = np.zeros(num_t_points)
        evals[i] = 1.0  # we are choosing the data points that are closest to our evaluation point
        l_i = P.fit(l_times, evals, est_d)

        # to checking that you built the right lagrange polynomial, evaluate it at the relevant points
        if verbose_lagrange:
            for j in range(num_t_points):
                print(f't = {l_times[j]:.3f}, l_i(t) = {l_i(l_times[j])}')

        # for every derivative that we estimate, compute this lagrange polynomial's derivative at the estimation time
        for q in range(est_d):
            l_bound[l_indices[i], q, delta-1] = l_i.deriv(q)(eval_time)  # coefficient for i-th residual in bound
            if verbose_lagrange:
                print(f'|l_{l_indices[i]}^({q})(t)|: {l_bound[l_indices[i], q, delta-1]}')


M = np.ones((ode_p,))
residual = np.zeros((ode_p, window_length))
cand_bounds = np.zeros((ode_p, est_d, deltas))
bounds = np.zeros((ode_p, est_d))
xhat_upper = np.zeros(ode_n)
xhat_lower = np.zeros(ode_n)

global_bounds = np.empty((ode_p, est_d))
for q in range(est_d):
    for r in range(ode_p):
        global_bounds[r, q] = (M[r]/(np.math.factorial(est_d+1)))*(np.sqrt(window_length**2+window_length))*((window_length*sampling_dt)**(est_d+1))
        global_bounds *= np.max(l_bound[:, q])
        global_bounds[r, q] += (M[r]/(np.math.factorial(est_d-q+1)))*(((q+1)*sampling_dt)**(est_d-q+1))
        comb = np.math.factorial(est_d)//(np.math.factorial(est_d-q+1)*np.math.factorial(max(0, q-1)))


def blit_rotate_center(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


# Sprite class
class CarSprite(pygame.sprite.Sprite):
    def __init__(self, center):
        super().__init__()
        self.state = np.zeros(ode_n)
        self.center = center
        self.image = self.create_image()
        self.rect = self.image.get_rect(center=self.center)
        self.state[0] = self.rect.x + self.image.get_rect().width/2
        self.state[1] = self.rect.y + self.image.get_rect().height/2

        self.op_data = np.vstack((np.ones(window_length, dtype=np.float32)*self.state[0], np.ones(window_length, dtype=np.float32)*self.state[1]))
        self.yhat = np.zeros((2, est_d+1))
        self.xhat = np.zeros(ode_n)
        self.u = np.zeros((2, 1))

    def create_image(self):
        max_width = 2*car_width
        max_height = 2*car_len
        image = pygame.Surface((max_width, max_height), pygame.SRCALPHA)
        pygame.draw.rect(image, BLUE, (car_width/2, car_len/2, car_width, car_len))

        wheel_fl = pygame.Surface((wheel_dep, wheel_diam), pygame.SRCALPHA)
        wheel_fl.fill(BLACK)
        blit_rotate_center(image, wheel_fl, (car_width - wheel_sep/2. - wheel_dep/2., car_len - axle_sep/2 - wheel_diam/2), math.degrees(self.state[4]))

        wheel_fr = pygame.Surface((wheel_dep, wheel_diam), pygame.SRCALPHA)
        wheel_fr.fill(BLACK)
        blit_rotate_center(image, wheel_fr, (car_width + wheel_sep/2. - wheel_dep/2., car_len - axle_sep/2 - wheel_diam/2), math.degrees(self.state[4]))

        pygame.draw.rect(image, BLACK, (car_width - wheel_sep/2. - wheel_dep/2., car_len + axle_sep/2 - wheel_diam/2, wheel_dep, wheel_diam))
        pygame.draw.rect(image, BLACK, (car_width + wheel_sep/2. - wheel_dep/2., car_len + axle_sep/2 - wheel_diam/2, wheel_dep, wheel_diam))
        return image

    def update(self, keys):
        self.u[:, :] = 0
        if keys[pygame.K_UP]:
            if self.state[3] < speed_limit:
                self.u[0, 0] = 0.1
        if keys[pygame.K_DOWN]:
            if self.state[3] > 0:
                self.u[0, 0] = -0.1
        if keys[pygame.K_LEFT]:
            if self.state[4] < math.radians(steering_limit):
                self.u[1, 0] = math.radians(2)
        if keys[pygame.K_RIGHT]:
            if self.state[4] > -math.radians(steering_limit):
                self.u[1, 0] = -math.radians(2)

        self.state[3] += self.u[0, 0]
        self.state[4] += self.u[1, 0]
        self.op_data = np.roll(self.op_data, -1)

        self.image = pygame.transform.rotate(self.create_image(), math.degrees(self.state[2]))
        self.rect = self.image.get_rect(center=self.rect.center)

        self.state[0] -= self.state[3] * math.sin(self.state[2])
        self.state[1] -= self.state[3] * math.cos(self.state[2])
        self.state[2] += self.state[3] * math.tan(self.state[4]) / axle_sep

        self.rect.x = self.state[0] - self.image.get_rect().width/2
        self.rect.y = self.state[1] - self.image.get_rect().height/2

        # Poly Estimator
        self.op_data[0, -1] = self.state[0]
        self.op_data[1, -1] = self.state[1]
        poly_estimator.fit(self.op_data)
        residual = poly_estimator.residuals

        for i in range(est_d+1):
            self.yhat[:, i] = poly_estimator.differentiate(eval_time, i)

        self.xhat = ODE.invert_position(0, self.yhat, self.u)
        self.xhat[3] /= FPS
        # if self.state[3] < 0:
        #     self.xhat[3] *= -1
        self.xhat[2] = -math.pi/2 - self.xhat[2]
        self.xhat[4] *= -1

        # compute a bound on derivative estimation error from residuals
        for q in range(est_d):
            for r in range(ode_p):
                noise_vector = np.ones(window_length,)*noise_mag
                # noise_vector = np.abs(noise_samples[r, t-N+1:t+1])
                for delta in range(deltas):
                    cand_bounds[r, q, delta] = np.abs(np.dot(residual[r, :], l_bound[:, q, delta]))
                    cand_bounds[r, q, delta] += np.dot(noise_vector, np.abs(l_bound[:, q, delta]))
                    cand_bounds[r, q, delta] += M[r]*comb*(((delta+1)*sampling_dt)**(est_d-q+1))
        bounds = np.min(cand_bounds, axis=-1)

        xhat_upper[0] = self.xhat[0] + bounds[0, 0]
        xhat_lower[0] = self.xhat[0] - bounds[0, 0]

        xhat_upper[1] = self.xhat[1] + bounds[1, 0]
        xhat_lower[1] = self.xhat[1] - bounds[1, 0]

        xhat_upper[2] = -math.pi/2 - np.arctan2(self.yhat[1, 1] + bounds[1, 1], self.yhat[0, 1] - bounds[0, 1])
        xhat_lower[2] = -math.pi/2 - np.arctan2(self.yhat[1, 1] - bounds[1, 1], self.yhat[0, 1] + bounds[0, 1])

        xhat_upper[3] = self.xhat[3] + np.linalg.norm(bounds[:2, 1])/FPS/3
        xhat_lower[3] = self.xhat[3] - np.linalg.norm(bounds[:2, 1])/FPS/3

        # this step does an exhaustive search through upper and lower bound combinations for the last component
        # it's not ideal but requires 16 computations in this case
        yhat_poly_ul = [self.yhat[:, :3] - bounds[:, :], self.yhat[:, :3] + bounds[:, :]]
        lb = np.inf
        ub = -np.inf
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        test = self.yhat[:, :].copy()
                        test[0, 1] = yhat_poly_ul[i][0, 1].copy()
                        test[1, 1] = yhat_poly_ul[j][1, 1].copy()
                        test[0, 2] = yhat_poly_ul[k][0, 2].copy()
                        test[1, 2] = yhat_poly_ul[l][1, 2].copy()
                        val = ODE.invert_position(0.0, test, self.u[:, :])[4]
                        lb = min(lb, val)
                        ub = max(ub, val)
        xhat_lower[4] = lb
        xhat_upper[4] = ub
        # i = 4
        # print(self.xhat[i], xhat_lower[i], xhat_upper[i])


def draw_text(text_string, top, left):
    txt_surf = font.render(text_string, True, BLACK)
    txt_rect = txt_surf.get_rect()
    txt_rect.top = top
    txt_rect.left = left
    screen.blit(txt_surf, txt_rect)


def draw_speed_bar(speed, speed_est, speed_lower, speed_upper):
    BAR_TOP = 10
    BAR_LEFT = 20
    BAR_LEN = 200
    BAR_WIDTH = 20
    TXT_OFFSET = 5
    SPEED_MAX = 10
    speed_pos = speed/SPEED_MAX*BAR_LEN+BAR_LEFT
    speed_pos_est = speed_est/SPEED_MAX*BAR_LEN+BAR_LEFT
    speed_lower_pos = speed_lower/SPEED_MAX*BAR_LEN+BAR_LEFT
    speed_width = speed_upper/SPEED_MAX*BAR_LEN+BAR_LEFT - max(BAR_LEFT, speed_lower_pos)
    pygame.draw.rect(screen, YELLOW, (max(BAR_LEFT, speed_lower_pos), BAR_TOP, speed_width, BAR_WIDTH))
    pygame.draw.rect(screen, BLACK, (BAR_LEFT, BAR_TOP, BAR_LEN, BAR_WIDTH), width=1)
    # pygame.draw.line(screen, GREY, (BAR_LEFT+BAR_LEN/2, BAR_TOP), (BAR_LEFT+BAR_LEN/2, BAR_TOP+BAR_WIDTH))
    pygame.draw.line(screen, RED, (speed_pos, BAR_TOP), (speed_pos, BAR_TOP+BAR_WIDTH), width=2)
    pygame.draw.line(screen, GREEN, (speed_pos_est, BAR_TOP), (speed_pos_est, BAR_TOP+BAR_WIDTH), width=2)
    draw_text(str(0), BAR_TOP+BAR_WIDTH+TXT_OFFSET, BAR_LEFT-TXT_OFFSET)
    draw_text(str(SPEED_MAX), BAR_TOP+BAR_WIDTH+TXT_OFFSET, BAR_LEN+BAR_LEFT-TXT_OFFSET)
    # draw_text(str(0), BAR_TOP+BAR_WIDTH+TXT_OFFSET, BAR_LEFT+BAR_LEN/2-TXT_OFFSET)


def draw_steering_circle(angle, angle_est, angle_lower, angle_upper):
    CENTER_X = 120
    CENTER_Y = 160
    RAD = 80
    arrow_tip = (CENTER_X-RAD*math.sin(angle_est), CENTER_Y-RAD*math.cos(angle_est))
    # pygame.draw.circle(screen, BLACK, (CENTER_X, CENTER_Y), RAD, width=1)
    # pygame.draw.polygon(screen, YELLOW, [(CENTER_X, CENTER_Y), (arrow_tip[0]-RAD*math.tan(angle_bound)*math.cos(angle_est), arrow_tip[1]+RAD*math.tan(angle_bound)*math.sin(angle_est)), (arrow_tip[0]+RAD*math.tan(angle_bound)*math.cos(angle_est), arrow_tip[1]-RAD*math.tan(angle_bound)*math.sin(angle_est))])
    pygame.draw.polygon(screen, YELLOW, [(CENTER_X, CENTER_Y), (CENTER_X-RAD*math.sin(angle_lower), CENTER_Y-RAD*math.cos(angle_lower)), (CENTER_X-RAD*math.sin(angle_upper), CENTER_Y-RAD*math.cos(angle_upper))])
    pygame.draw.arc(screen, BLACK, ((CENTER_X-RAD, CENTER_Y-RAD), (2*RAD, 2*RAD)), 0, math.pi)
    pygame.draw.line(screen, BLACK, (CENTER_X-RAD, CENTER_Y), (CENTER_X+RAD, CENTER_Y))
    pygame.draw.line(screen, GREY, (CENTER_X, CENTER_Y), (CENTER_X, CENTER_Y-RAD))
    pygame.draw.line(screen, RED, (CENTER_X, CENTER_Y), (CENTER_X-RAD*math.sin(angle), CENTER_Y-RAD*math.cos(angle)), width=2)
    pygame.draw.line(screen, GREEN, (CENTER_X, CENTER_Y), arrow_tip, width=2)
    draw_text(str(0), CENTER_Y-RAD-15, CENTER_X-5)
    draw_text(str(90), CENTER_Y-10, CENTER_X+RAD+5)
    draw_text(str(-90), CENTER_Y-10, CENTER_X-RAD-25)


def draw_heading(sprite, angle_lower, angle_upper):
    # heading_est = sprite.xhat[2]
    # arrow_tip = (sprite.state[0]-ARROW_LEN*math.sin(heading_est), sprite.state[1]-ARROW_LEN*math.cos(heading_est))
    # pygame.draw.polygon(screen, YELLOW, [(sprite.state[0], sprite.state[1]), (arrow_tip[0]-ARROW_LEN*math.tan(angle_lower)*math.cos(heading_est), arrow_tip[1]+ARROW_LEN*math.tan(angle_lower)*math.sin(heading_est)), (arrow_tip[0]+ARROW_LEN*math.tan(angle_upper)*math.cos(heading_est), arrow_tip[1]-ARROW_LEN*math.tan(angle_upper)*math.sin(heading_est))])
    pygame.draw.polygon(screen, YELLOW, [(sprite.state[0], sprite.state[1]), (sprite.state[0]-2*ARROW_LEN*math.sin(angle_lower), sprite.state[1]-2*ARROW_LEN*math.cos(angle_lower)), (sprite.state[0]-2*ARROW_LEN*math.sin(angle_upper), sprite.state[1]-2*ARROW_LEN*math.cos(angle_upper))])
    # pygame.draw.polygon(screen, YELLOW, [(sprite.state[0], sprite.state[1]), (sprite.state[0]-ARROW_LEN*math.sin(angle_lower)/math.cos(heading_est-angle_lower), sprite.state[1]-ARROW_LEN*math.cos(angle_lower)/math.cos(heading_est-angle_lower)), (sprite.state[0]-ARROW_LEN*math.sin(angle_upper)/math.cos(heading_est-angle_upper), sprite.state[1]-ARROW_LEN*math.cos(angle_upper)/math.cos(heading_est-angle_upper))])


# Clock to control the frame rate
clock = pygame.time.Clock()

center_point = (WIDTH // 2, HEIGHT // 2)
sprite = CarSprite(center_point)
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
    draw_heading(sprite, xhat_lower[2], xhat_upper[2])
    all_sprites.draw(screen)
    pygame.draw.line(screen, RED, (sprite.state[0], sprite.state[1]), (sprite.state[0]-ARROW_LEN*math.sin(sprite.state[2]), sprite.state[1]-ARROW_LEN*math.cos(sprite.state[2])), width=2)
    pygame.draw.line(screen, GREEN, (sprite.state[0], sprite.state[1]), (sprite.state[0]-ARROW_LEN*math.sin(sprite.xhat[2]), sprite.state[1]-ARROW_LEN*math.cos(sprite.xhat[2])), width=2)
    # pygame.draw.circle(screen, RED, (sprite.xhat[0], sprite.xhat[1]), 10)
    # draw_speed_bar(sprite.state[3])

    draw_speed_bar(sprite.state[3], sprite.xhat[3], xhat_lower[3], xhat_upper[3])
    draw_steering_circle(sprite.state[4], sprite.xhat[4], xhat_lower[4], xhat_upper[4])

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
