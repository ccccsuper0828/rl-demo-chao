"""
Dino Jump Game - Version 5.0
Based on insights from open source projects:
- https://github.com/aome510/chrome-dino-game-rl
- https://github.com/hfahrudin/trex-DQN
"""

import pygame
import random
import numpy as np
from .constants import *


class Dino:
    """Player character"""

    def __init__(self):
        self.x = DINO_X
        self.y = GROUND_Y - DINO_HEIGHT
        self.width = DINO_WIDTH
        self.height = DINO_HEIGHT
        self.velocity_y = 0
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_VELOCITY
            self.is_jumping = True

    def update(self):
        self.velocity_y += GRAVITY
        self.y += self.velocity_y

        if self.y >= GROUND_Y - DINO_HEIGHT:
            self.y = GROUND_Y - DINO_HEIGHT
            self.velocity_y = 0
            self.is_jumping = False

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        rect = self.get_rect()
        pygame.draw.rect(screen, GRAY, rect)
        eye_x = rect.x + rect.width - 10
        eye_y = rect.y + 8
        pygame.draw.circle(screen, WHITE, (eye_x, eye_y), 4)
        pygame.draw.circle(screen, BLACK, (eye_x + 1, eye_y), 2)


class Obstacle:
    """Obstacle"""

    def __init__(self, x: float, speed: float):
        self.x = x
        self.speed = speed
        self.width = OBSTACLE_WIDTH
        self.height = random.randint(OBSTACLE_MIN_HEIGHT, OBSTACLE_MAX_HEIGHT)
        self.y = GROUND_Y - self.height
        self.passed = False

    def update(self):
        self.x -= self.speed

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.get_rect())


class DinoGame:
    """
    Game Environment - Version 6.1 (Clean v5.0)

    Key insight from aome510/chrome-dino-game-rl:
    "The reward is defined to be the number of obstacles that the agent passes"
    Simple reward = better learning
    """

    def __init__(self, render: bool = True):
        self.render_game = render

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Dino Jump - Q-Learning v6.1 (Clean)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self):
        self.dino = Dino()
        self.obstacles = []
        self.score = 0
        self.speed = OBSTACLE_SPEED_INIT
        self.game_over = False
        self.frames_survived = 0
        self.obstacles_passed = 0

        self._spawn_obstacle()
        return self.get_state()

    def _spawn_obstacle(self):
        if len(self.obstacles) == 0:
            x = WINDOW_WIDTH
        else:
            last_x = max(obs.x for obs in self.obstacles)
            gap = random.randint(OBSTACLE_GAP_MIN, OBSTACLE_GAP_MAX)
            x = last_x + gap

        obstacle = Obstacle(x, self.speed)
        self.obstacles.append(obstacle)

    def get_state(self):
        """
        State representation (6 dimensions).

        0. Distance to nearest obstacle  (normalized 0-1)
        1. Urgency signal                (1 = close, 0 = far)
        2. Is jumping                    (0 or 1)
        3. Vertical velocity             (normalized)
        4. Current obstacle speed         (normalized 0-1)
        5. Nearest obstacle height        (normalized 0-1)

        Speed and height are needed so the agent can learn
        speed-dependent optimal jump timing.
        """
        state = np.zeros(6, dtype=np.float32)

        # Find nearest obstacle ahead of (or overlapping) the Dino
        nearest_obs = None
        for obs in sorted(self.obstacles, key=lambda o: o.x):
            if obs.x + obs.width > self.dino.x:
                nearest_obs = obs
                break

        if nearest_obs:
            # Distance to obstacle (normalized 0-1, where 0 = very close)
            dist = nearest_obs.x - (self.dino.x + self.dino.width)
            state[0] = max(0, min(1, dist / 400))

            # Urgency signal: 1 when close, 0 when far
            state[1] = max(0, 1 - dist / 150) if dist > 0 else 1.0

            # Obstacle height (normalized by max height)
            state[5] = nearest_obs.height / OBSTACLE_MAX_HEIGHT
        else:
            state[0] = 1.0
            state[1] = 0.0
            state[5] = 0.0

        # Is jumping (0 or 1)
        state[2] = 1.0 if self.dino.is_jumping else 0.0

        # Vertical velocity (normalized)
        state[3] = self.dino.velocity_y / 20.0

        # Current obstacle speed (normalized 0-1)
        state[4] = self.speed / OBSTACLE_SPEED_MAX

        return state

    def _get_nearest_obstacle(self):
        """Get (distance, height) to nearest obstacle ahead of Dino.

        Returns (float('inf'), 0) when there is no obstacle ahead.
        """
        dino_right = self.dino.x + self.dino.width
        best_dist = float('inf')
        best_height = 0
        for obs in self.obstacles:
            if obs.x + obs.width > self.dino.x:
                dist = obs.x - dino_right
                if dist < best_dist:
                    best_dist = dist
                    best_height = obs.height
        return best_dist, best_height

    def step(self, action: int):
        """Execute one game step"""
        # Track whether a new jump was initiated this frame
        was_grounded = not self.dino.is_jumping

        # Action: 0 = don't jump, 1 = jump
        if action == 1:
            self.dino.jump()

        # Detect if a new jump just started
        jumped_this_frame = was_grounded and self.dino.is_jumping

        self.dino.update()

        # Update obstacles
        for obs in self.obstacles:
            obs.speed = self.speed
            obs.update()

        # Check passed obstacles
        passed = 0
        for obs in self.obstacles:
            if not obs.passed and obs.x + obs.width < self.dino.x:
                obs.passed = True
                passed += 1

        # Remove off-screen
        self.obstacles = [obs for obs in self.obstacles if obs.x > -100]

        # Update score
        if passed > 0:
            self.obstacles_passed += passed
            self.score += passed * 10

        # Spawn new obstacles
        while len(self.obstacles) < 2:
            self._spawn_obstacle()

        # Check collision
        collision = False
        dino_rect = self.dino.get_rect()
        for obs in self.obstacles:
            if dino_rect.colliderect(obs.get_rect()):
                collision = True
                self.game_over = True
                break

        # Reward with jump penalty awareness
        reward = self._calculate_reward(collision, passed, jumped_this_frame)

        # Update speed
        self.speed = min(OBSTACLE_SPEED_MAX,
                        OBSTACLE_SPEED_INIT + self.frames_survived * SPEED_INCREMENT)

        self.frames_survived += 1

        if self.render_game:
            self._render()

        return self.get_state(), reward, self.game_over, {
            "score": self.score,
            "frames": self.frames_survived,
            "obstacles_passed": self.obstacles_passed
        }

    # ---- jump-timing physics constants ----
    # From JUMP_VELOCITY=-18, GRAVITY=1.2, DINO_HEIGHT=50:
    #   Dino bottom at frame f = 320 - 17.4f + 0.6f²
    #   Safe when bottom < GROUND_Y - obs_height
    #   → 0.6f² - 17.4f + obs_height < 0
    # Solved via quadratic formula for continuous, height-dependent windows.
    _DANGER_ZONE = DINO_WIDTH + OBSTACLE_WIDTH   # 60 px

    def _jump_timing_quality(self, dist, obs_height):
        """Return a value in [-1, 1] rating the jump timing.

        Uses continuous physics to compute the exact safe-frame window
        for the given *obs_height*, then adds a height-proportional
        safety margin so that taller obstacles require more precise
        timing (tighter window).

        Physics (quadratic):
          0.6 f² − 17.4 f + h < 0
          f = (17.4 ± √(302.76 − 2.4 h)) / 1.2

        Safety margin (frames, each side):
          margin = h / 25          (h=30 → 1.2,  h=50 → 2.0)

        Returns:
          +1.0  at optimal distance
           0.0  at the edges of the safe window
          -1.0  far outside the window
        """
        import math

        h = max(obs_height, 1)   # avoid division by zero
        discriminant = 302.76 - 2.4 * h
        if discriminant <= 0:
            return -1.0          # obstacle too tall to ever clear

        sqrt_d = math.sqrt(discriminant)
        f_first_raw = (17.4 - sqrt_d) / 1.2   # earliest safe frame
        f_last_raw  = (17.4 + sqrt_d) / 1.2   # latest safe frame

        # Height-proportional safety margin (in frames).
        # Taller obstacles → bigger margin → tighter window.
        margin = h / 25.0        # h=30→1.2  h=40→1.6  h=50→2.0

        first_safe = f_first_raw + margin
        last_safe  = f_last_raw  - margin

        if last_safe <= first_safe:
            return -1.0

        # Convert frame window to distance window
        d_min = first_safe * self.speed
        d_max = last_safe  * self.speed - self._DANGER_ZONE
        if d_max <= d_min:
            d_max = d_min + 1

        d_optimal   = (d_min + d_max) / 2.0
        half_window = (d_max - d_min) / 2.0
        if half_window < 1:
            half_window = 1.0

        # Normalised error: 0 at optimal, 1 at edges, >1 outside
        error = abs(dist - d_optimal) / half_window

        # Smooth quality: 1 → 0 → −1
        quality = 1.0 - error
        return max(-1.0, min(1.0, quality))

    def _calculate_reward(self, collision, passed, jumped_this_frame=False):
        """
        Reward function with physics-based jump-timing feedback.

        Design principles (informed by DinoRunTutorial reference):
        - Main signal: +1 pass, -1 death          (sparse but clear)
        - Jump timing: up to +0.4 for optimal timing (dense shaping)
        - Unnecessary jump: -0.3                    (prevents spam)
        - Survival: +0.01 per frame                 (small baseline)
        """
        if collision:
            return -1.0   # Death penalty

        if passed > 0:
            return 1.0    # Reward for passing obstacle

        if jumped_this_frame:
            dist, obs_height = self._get_nearest_obstacle()

            # No obstacle ahead at all → clearly unnecessary
            if dist > 400:
                return -0.3

            quality = self._jump_timing_quality(dist, obs_height)

            if quality > 0:
                # Within safe window → reward proportional to quality
                return 0.4 * quality          # max +0.4 at optimal
            else:
                # Outside safe window → penalty proportional to badness
                return 0.3 * quality          # max -0.3 at far edge

        # Small survival bonus
        return 0.01

    def _render(self):
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, GROUND_Y), (WINDOW_WIDTH, GROUND_Y), 2)
        self.dino.draw(self.screen)
        for obs in self.obstacles:
            obs.draw(self.screen)

        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (WINDOW_WIDTH - 150, 20))

        if self.game_over:
            game_over_text = self.font.render("GAME OVER - Press R to restart", True, RED)
            text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()
        self.clock.tick(FPS)

    def handle_human_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_ESCAPE:
                    return False, None

        keys = pygame.key.get_pressed()
        action = 1 if keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w] else 0
        return True, action

    def close(self):
        if self.render_game:
            pygame.quit()


if __name__ == "__main__":
    game = DinoGame(render=True)
    running = True
    print("Controls: SPACE/UP/W = Jump, R = Restart, ESC = Quit")

    while running:
        running, action = game.handle_human_input()
        if running and action is not None:
            state, reward, done, info = game.step(action)
            if done:
                print(f"Game Over! Score: {info['score']}")

    game.close()
