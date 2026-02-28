"""
Game Constants - Version 5.0
Tuned based on open source project insights
"""

# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (83, 83, 83)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# Dino settings
DINO_WIDTH = 40
DINO_HEIGHT = 50
DINO_X = 80
GROUND_Y = WINDOW_HEIGHT - 80

# Jump physics - tuned for faster jumps
# Insight from hfahrudin/trex-DQN: shorter air time = more learning opportunities
JUMP_VELOCITY = -18  # Increased from -15 for faster jump
GRAVITY = 1.2        # Increased from 0.8 for faster fall

# Obstacle settings - easier for learning
OBSTACLE_WIDTH = 20
OBSTACLE_MIN_HEIGHT = 30
OBSTACLE_MAX_HEIGHT = 50
OBSTACLE_SPEED_INIT = 5
OBSTACLE_SPEED_MAX = 10
OBSTACLE_GAP_MIN = 400
OBSTACLE_GAP_MAX = 600

# Game settings
SPEED_INCREMENT = 0.0005  # Slower speed increase
