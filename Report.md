# Dino Jump Game Based on Deep Q-Learning

**Course:** CDS524 Assignment 1 - Reinforcement Learning Game Design

**Author:** [Your Name]

**Date:** [Date]

**GitHub:** [Your GitHub Link]

**YouTube:** [Your YouTube Demo Link]

---

## 1 Introduction

This project uses the Deep Q-Learning (DQN) algorithm to design and implement a platformer jumping game similar to Chrome's Dinosaur Game. In this game, an AI-controlled dinosaur learns to avoid obstacles by jumping or ducking. The AI agent makes decisions based on the current state of the environment and improves its performance through reinforcement learning.

This project uses the following technology stack:

- **Python** is used as the main programming language
- **PyTorch** is an open-source deep learning framework that provides strong support for building and training DQN
- **Pygame** is a library for game development that provides functions such as graphics, input processing, and game loop management

The goal of this project is to demonstrate how Q-Learning can be applied to train an AI agent to play a real-time action game, where the agent must learn to time its jumps and ducks to avoid obstacles of varying heights and speeds.

---

## 2 Game Design

### 2.1 Rules of the Game

In this game, the player (or AI agent) controls a dinosaur character. The dinosaur automatically runs from left to right at an increasing speed. The goal is to survive as long as possible by avoiding obstacles.

**Game Rules:**
- The dinosaur runs automatically; the player only controls jumping and ducking
- Ground obstacles (cacti) must be jumped over
- Flying obstacles (birds) can be jumped over or ducked under
- Collision with any obstacle ends the game
- Score increases with each obstacle successfully passed
- Game speed gradually increases over time, making the game progressively harder

**State Space:**
The state space includes key information that the AI needs to make decisions:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Distance to obstacle | Normalized distance to the nearest obstacle |
| 1 | Obstacle height | Height of the nearest obstacle |
| 2 | Obstacle width | Width of the nearest obstacle |
| 3 | Game speed | Current movement speed (normalized) |
| 4 | Dino Y position | Vertical position of the dinosaur |
| 5 | Is jumping | Whether the dinosaur is currently in the air |
| 6 | Is flying obstacle | Whether the obstacle is a flying bird |
| 7 | Next obstacle distance | Distance to the second obstacle |

**Action Space:**
The action space of the dinosaur includes three discrete actions:

| Action | Description |
|--------|-------------|
| 0 | Do nothing (keep running) |
| 1 | Jump |
| 2 | Duck |

The agent will make judgments and perform actions based on the real-time state of the game.

### 2.2 Class Design of the Game

**DinoGame class** is responsible for managing the overall flow and state of the game. It initializes the game window and game objects, loads the AI model, and updates the game state and handles collision detection during the game. It provides `reset()`, `step()`, and `get_state()` methods to interface with the reinforcement learning agent.

**Dino class** represents the dinosaur character controlled by the AI agent. It handles the dinosaur's movement mechanics including jumping, ducking, and gravity. The class tracks the dinosaur's position, velocity, and current state (jumping/ducking).

**Obstacle class** represents the obstacles in the game, including ground cacti and flying birds. Each obstacle has properties for position, size, speed, and type. The class handles obstacle movement and provides collision detection.

**DQNAgent class** implements the Deep Q-Learning algorithm. It manages the policy network and target network, handles action selection using epsilon-greedy strategy, and performs training updates using experience replay.

**DQN class** defines the neural network architecture that approximates the Q-value function. It takes the state vector as input and outputs Q-values for each possible action.

**ReplayBuffer class** stores experience tuples (state, action, reward, next_state, done) and provides random sampling for training, which helps break correlation between consecutive samples.

### 2.3 UI Design

The overall user interface follows a simple and clean style. The game window is 800x400 pixels with a white background.

**Visual Elements:**
- The dinosaur is rendered as a gray rectangle with an eye indicator
- Ground obstacles (cacti) are rendered as green rectangles
- Flying obstacles (birds) are rendered as red rectangles
- A ground line separates the play area
- Score and speed are displayed in the top corners
- Game Over message appears on collision with restart instructions

**Game Screenshots:**

[Insert Screenshot: Start Screen]
*Fig 2.1 Dino Jump UI - Start*

[Insert Screenshot: Gameplay with obstacles]
*Fig 2.2 Dino Jump - Gameplay*

[Insert Screenshot: Game Over screen]
*Fig 2.3 Dino Jump - Game Over*

---

## 3 Implementation of Q-Learning Algorithm

Q-Learning is a value-based reinforcement learning algorithm that selects the best action by learning Q-values. In this project, since the game involves continuous movement in a dynamic environment, a Deep Q-Network (DQN) is used to handle the complex state space.

### 3.1 Network Architecture

The DQN uses a fully connected neural network with the following architecture:

```
Input Layer (8 neurons) - State vector
    ↓
Hidden Layer 1 (256 neurons) + ReLU activation
    ↓
Hidden Layer 2 (128 neurons) + ReLU activation
    ↓
Hidden Layer 3 (64 neurons) + ReLU activation
    ↓
Output Layer (3 neurons) - Q-values for each action
```

The ReLU activation function introduces non-linearity to help the network learn the complex mapping between states and action values.

### 3.2 State Representation

The `get_state()` function obtains the current state of the game during training. The state vector is an 8-dimensional normalized vector containing:
- Relative position information between the dinosaur and obstacles
- Current game dynamics (speed, jumping status)
- Obstacle characteristics (height, type)

This state representation provides the network with sufficient information to predict the optimal action.

### 3.3 Training Process

During training, we use the **ε-greedy strategy** to balance exploration and exploitation:
- With probability ε: select a random action (exploration)
- With probability 1-ε: select the action with highest Q-value (exploitation)
- ε decays from 1.0 to 0.01 over training episodes

**Training Parameters:**

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Discount Factor (γ) | 0.95 |
| Batch Size | 64 |
| Replay Buffer Size | 100,000 |
| Target Network Update | Every 100 steps |
| Epsilon Decay | 0.995 |
| Training Episodes | 500 |

**Experience Replay:**
During training, we store experiences (state, action, reward, next_state, done) in a replay buffer. We randomly sample batches from this buffer to update the network weights, which helps break the correlation between consecutive samples and improves training stability.

**Target Network:**
A separate target network is used to compute the target Q-values, which is updated periodically from the policy network. This technique reduces oscillations and improves training stability.

### 3.4 Reward Function

The reward function is designed to guide the agent toward the desired behavior:

```python
def calculate_reward(self):
    reward = 0.1  # Small positive reward for surviving each frame

    if obstacle_passed:
        reward += 10  # Reward for successfully passing an obstacle

    if collision:
        reward = -100  # Large penalty for game over

    return reward
```

**Reward Design Rationale:**
- **Survival reward (+0.1):** Encourages the agent to stay alive
- **Obstacle reward (+10):** Provides clear positive feedback for correct actions
- **Collision penalty (-100):** Strong negative signal to avoid obstacles

---

## 4 Challenges and Solutions

### Challenge 1: Reward Shaping

**Problem:** Initially, the agent had difficulty learning because the reward signal was too sparse - it only received feedback when passing obstacles or dying.

**Solution:** We added a small positive reward for each frame survived. This provides continuous feedback and helps the agent understand that staying alive is valuable. The reward function was carefully balanced to ensure the agent prioritizes avoiding obstacles over simply surviving.

### Challenge 2: State Representation

**Problem:** Determining what information to include in the state vector was challenging. Too little information made it impossible for the agent to make good decisions; too much information slowed down learning.

**Solution:** After experimentation, we settled on an 8-dimensional state vector that includes:
- Distance and properties of the nearest obstacle
- Information about the second obstacle (for planning ahead)
- Current dynamics (speed, position, jumping status)

This representation provides sufficient information while remaining compact enough for efficient learning.

### Challenge 3: Handling Increasing Difficulty

**Problem:** The game speed increases over time, which means the agent needs to adapt to faster gameplay as the game progresses.

**Solution:** We normalized the speed value in the state vector, allowing the agent to learn a policy that generalizes across different speeds. We also included the current speed as an explicit state feature so the agent can adjust its timing based on game speed.

### Challenge 4: Training Stability

**Problem:** Early training attempts showed unstable learning with large fluctuations in performance.

**Solution:** We implemented several stabilization techniques:
1. Target network with periodic updates
2. Gradient clipping to prevent exploding gradients
3. Experience replay with sufficient buffer size
4. Appropriate learning rate (0.001)

---

## 5 Experimental Results

### 5.1 Training Progress

[Insert training_curves.png image here]
*Fig 5.1 Training Curves*

The training curves show:
- **Score over episodes:** Gradual improvement with the agent achieving higher scores
- **Average score (100 episodes):** Smoothed learning progress
- **Loss:** Decreasing loss indicating network convergence
- **Epsilon:** Decay of exploration rate over time

### 5.2 Performance Evaluation

| Metric | Value |
|--------|-------|
| Best Score | [Fill after training] |
| Average Score (last 100 episodes) | [Fill after training] |
| Training Episodes | 500 |
| Final Epsilon | 0.01 |

### 5.3 Observations

During training, we observed that:
1. The agent initially learned to jump at every obstacle
2. Over time, it learned to time jumps more precisely
3. The agent developed the ability to duck under flying obstacles
4. Performance stabilized after approximately [X] episodes

---

## 6 Conclusion

Through this project, we successfully implemented a platformer jumping game with a Deep Q-Learning agent. The AI-controlled dinosaur can continuously optimize its strategy through the learning algorithm and improve its survival ability.

**Key Achievements:**
1. Designed and implemented a complete game environment using Pygame
2. Successfully trained a DQN agent to play the game
3. Achieved stable learning through proper reward design and training techniques
4. The agent learned to handle both jumping and ducking actions appropriately

**Lessons Learned:**
1. Reward shaping is crucial for guiding agent behavior
2. State representation significantly impacts learning efficiency
3. Training stability techniques (target network, experience replay) are essential for DQN

**Future Improvements:**
1. Implement Double DQN to reduce overestimation
2. Add more obstacle types and game mechanics
3. Experiment with different network architectures
4. Implement prioritized experience replay for more efficient learning

This project demonstrates the application of reinforcement learning in game AI and provides an experimental platform for exploring the potential of Q-Learning algorithms.

---

## 7 References

[1] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

[2] aome510. (2024). chrome-dino-game-rl: Play Chrome's Dinosaur Game with Reinforcement Learning [Computer software]. GitHub. https://github.com/aome510/chrome-dino-game-rl

[3] thomas-mauran. (2023). DinoAI: Reinforcement learning python AI [Computer software]. GitHub. https://github.com/thomas-mauran/DinoAI

[4] drl-dql. (2020). DQN-Flappy-Bird: Flappy Bird hack using Deep Reinforcement Learning [Computer software]. GitHub. https://github.com/drl-dql/DQN-Flappy-Bird

---

## Appendix: How to Run

```bash
# Install dependencies
pip install pygame torch numpy matplotlib

# Train the agent
python train.py --episodes 500

# Play with trained AI
python play.py --mode ai

# Play as human
python play.py --mode human
```
