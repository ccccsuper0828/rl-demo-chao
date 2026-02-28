# Dino Jump - Deep Q-Learning Platformer Game

## CDS524 Assignment 1 - Reinforcement Learning Game Design

A platformer game where an AI agent learns to play using Deep Q-Learning, inspired by Chrome's Dinosaur Game.

---

## Project Structure

```
Assignment1_RL_Game/
├── game/
│   ├── __init__.py
│   ├── dino_game.py       # Game environment (Pygame)
│   └── constants.py       # Game constants
├── agent/
│   ├── __init__.py
│   ├── dqn_model.py       # Deep Q-Network (PyTorch)
│   ├── agent.py           # DQN Agent
│   └── replay_buffer.py   # Experience Replay
├── model/                 # Saved models
├── train.py               # Training script
├── play.py                # Play with AI or human
├── DinoJump_QLearning.ipynb  # Jupyter notebook
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the AI Agent
```bash
python train.py --episodes 500
python train.py --episodes 500 --render  # Watch training
```

### Play the Game
```bash
# AI plays
python play.py --mode ai --model model/best_model.pth

# Human plays
python play.py --mode human

# Compare AI vs Human
python play.py --mode compare
```

### Human Controls
- **SPACE / UP / W** - Jump
- **DOWN / S** - Duck
- **R** - Restart
- **ESC** - Quit

---

## Game Design

### Rules
- The dinosaur runs automatically from left to right
- Obstacles (cacti and birds) appear from the right
- Jump over ground obstacles, duck under flying obstacles
- Game ends on collision
- Score increases with obstacles passed

### State Space (8 dimensions)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Distance | Distance to nearest obstacle |
| 1 | Height | Obstacle height |
| 2 | Width | Obstacle width |
| 3 | Speed | Current game speed |
| 4 | Dino Y | Dinosaur Y position |
| 5 | Jumping | Is dinosaur jumping (0/1) |
| 6 | Flying | Is obstacle flying (0/1) |
| 7 | Next Dist | Distance to second obstacle |

### Action Space (3 actions)
| Action | Description |
|--------|-------------|
| 0 | Do nothing (run) |
| 1 | Jump |
| 2 | Duck |

### Reward Function
| Event | Reward |
|-------|--------|
| Survive each frame | +0.1 |
| Pass an obstacle | +10 |
| Collision (game over) | -100 |

---

## Q-Learning Implementation

### DQN Architecture
```
Input (8) → FC(256) → ReLU → FC(128) → ReLU → FC(64) → ReLU → Output (3)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Discount Factor (γ) | 0.95 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.01 |
| Epsilon Decay | 0.995 |
| Batch Size | 64 |
| Buffer Size | 100,000 |
| Target Update | Every 100 steps |

### Key Techniques
1. **Experience Replay** - Break correlation between samples
2. **Target Network** - Stabilize training
3. **Epsilon-Greedy** - Balance exploration and exploitation

---

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
2. [Chrome Dino Game RL](https://github.com/aome510/chrome-dino-game-rl)
3. [DQN Flappy Bird](https://github.com/drl-dql/DQN-Flappy-Bird)
4. [DinoAI](https://github.com/thomas-mauran/DinoAI)

---

## Deliverables Checklist

- [ ] Google Colab / Jupyter Notebook with code
- [ ] Written report (1000-1500 words)
- [ ] GitHub repository
- [ ] YouTube demo video
- [ ] All files in "Assignment 1 - [Your Name]" folder

**Deadline:** March 3, 2026, 8:00 PM
# rl-demo-chao
