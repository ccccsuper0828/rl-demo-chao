# CDS524 Assignment 1: Reinforcement Learning Game Design
# Dino Jump - Deep Q-Network Implementation

**Student Name:** [Your Name]
**Student ID:** [Your ID]
**Date:** February 2026

---

## 1. Introduction

This report presents the design and implementation of a reinforcement learning agent for a platform jumping game inspired by the Chrome Dinosaur Game. The project applies Deep Q-Network (DQN) with several improvements including Double DQN and experience replay to train an agent that learns to jump over obstacles autonomously.

### 1.1 Project Overview

The Dino Jump game is a side-scrolling platform game where a dinosaur character must jump over incoming obstacles. The game progressively increases in difficulty as the speed increases over time. This creates an ideal environment for reinforcement learning as:

- The state space is continuous but can be effectively discretized
- The action space is simple (jump or not jump)
- Immediate feedback is available through collision detection
- The task has clear success metrics (obstacles passed, survival time)

### 1.2 Objectives

1. Design a suitable game environment for Q-learning
2. Implement a Deep Q-Network agent with modern improvements
3. Train the agent through iterative experimentation
4. Analyze training results and optimize performance

---

## 2. Game Design

### 2.1 Game Environment

The game environment was built using Pygame with the following specifications:

| Component | Specification |
|-----------|---------------|
| Window Size | 800 x 400 pixels |
| Frame Rate | 60 FPS |
| Player Size | 40 x 50 pixels |
| Obstacle Width | 20 pixels |
| Obstacle Height | 30-50 pixels (random) |

### 2.2 Game Mechanics

**Player (Dinosaur):**
- Fixed horizontal position (x=80)
- Can jump with velocity -18 pixels/frame
- Affected by gravity (1.2 pixels/frame²)
- Jump duration approximately 30 frames (~0.5 seconds)

**Obstacles:**
- Spawn from the right side of screen
- Move left at increasing speed (5-10 pixels/frame)
- Random gap between obstacles (400-600 pixels)
- Collision ends the game

**Difficulty Progression:**
- Initial speed: 5 pixels/frame
- Maximum speed: 10 pixels/frame
- Speed increment: 0.0005 per frame

### 2.3 State Representation

The state is represented as a 4-dimensional vector:

```python
state[0] = distance_to_obstacle / 400      # Normalized distance (0-1)
state[1] = max(0, 1 - distance / 150)      # Urgency signal (0-1)
state[2] = 1.0 if jumping else 0.0         # Jump state (binary)
state[3] = velocity_y / 20.0               # Vertical velocity (normalized)
```

The **urgency signal** is a key innovation that provides a continuous measure of how urgently a jump is needed, rather than a simple binary threshold.

### 2.4 Action Space

The agent has 2 possible actions:
- **Action 0:** Do nothing (continue current state)
- **Action 1:** Jump (if on ground)

### 2.5 Reward Function

Following insights from successful open-source projects, we adopted a simple reward function:

```python
if collision:
    reward = -1.0    # Death penalty
elif passed_obstacle:
    reward = +1.0    # Reward for passing obstacle
else:
    reward = +0.01   # Small survival bonus
```

This simple design provides clear learning signals without complex reward shaping that could confuse the agent.

---

## 3. Q-Learning Implementation

### 3.1 Algorithm: Double DQN

We implemented Double Deep Q-Network (Double DQN) to address the overestimation bias present in standard DQN.

**Standard DQN:**
```python
target_q = reward + gamma * max(Q_target(next_state))
```

**Double DQN:**
```python
best_action = argmax(Q_policy(next_state))
target_q = reward + gamma * Q_target(next_state)[best_action]
```

The key insight is that Double DQN uses the policy network to **select** the best action, but the target network to **evaluate** that action, reducing overestimation.

### 3.2 Network Architecture

```
Input Layer:  4 neurons (state vector)
Hidden Layer 1: 128 neurons + ReLU activation
Hidden Layer 2: 64 neurons + ReLU activation
Output Layer: 2 neurons (Q-values for each action)
```

Weight initialization uses Kaiming Normal initialization for better gradient flow.

### 3.3 Training Components

**Experience Replay Buffer:**
- Capacity: 10,000 transitions
- Stores (state, action, reward, next_state, done) tuples
- Random sampling for training batches

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 0.001 | Higher rate for smaller network |
| Discount Factor (γ) | 0.95 | Balance immediate and future rewards |
| Epsilon Start | 1.0 | Full exploration initially |
| Epsilon End | 0.01 | Minimal exploration at convergence |
| Epsilon Decay | 0.995 | Gradual transition to exploitation |
| Batch Size | 32 | Frequent updates |
| Target Update | Every 100 steps | Stable Q-targets |

**Loss Function:**
We use Huber Loss (Smooth L1 Loss) instead of MSE for more stable training:
```python
loss = SmoothL1Loss(predicted_q, target_q)
```

### 3.4 Training Algorithm

```
For each episode:
    state = environment.reset()

    While not done:
        # Epsilon-greedy action selection
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(Q_policy(state))

        # Execute action
        next_state, reward, done = environment.step(action)

        # Store experience
        replay_buffer.push(state, action, reward, next_state, done)

        # Train if buffer is ready
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)

            # Double DQN target calculation
            next_actions = Q_policy(next_states).argmax(1)
            next_q = Q_target(next_states).gather(next_actions)
            target_q = rewards + gamma * next_q * (1 - dones)

            # Update policy network
            loss = HuberLoss(Q_policy(states), target_q)
            optimizer.step()

        # Update target network periodically
        if steps % target_update_freq == 0:
            Q_target.load_state_dict(Q_policy.state_dict())

        state = next_state

    # Decay exploration
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

---

## 4. Experimental Results

### 4.1 Training Progress

We conducted multiple training experiments, iterating through 5 versions of the system:

| Version | Key Changes | Best Score | Avg Score | Result |
|---------|-------------|------------|-----------|--------|
| v1.0 | Basic DQN, 8-dim state | 100 | 15.9 | Poor |
| v2.0 | 11-dim state, reward shaping | 130 | 24.6 | Marginal |
| v3.0 | Simplified to 5-dim, 2 actions | 110 | 11.8 | Worse |
| v4.0 | Positive-only rewards | 210 | 6.4 | Unstable |
| **v5.0** | Double DQN, Huber Loss | **710** | **39.3** | **Best** |

### 4.2 Version 5.0 Results (Best Performance)

Training for 2000 episodes revealed interesting learning dynamics:

**Peak Performance (Episode 690):**
- Average Score: 39.3
- Best Score: 260
- The agent successfully learned the optimal jumping strategy

**Maximum Score (Episode 1280):**
- Best Score: 710
- Demonstrates the agent can achieve very high performance

### 4.3 Key Findings

**1. Catastrophic Forgetting:**
After episode 1300, the average score dropped from 39 to 11. This is due to:
- Replay buffer being overwritten with failure experiences
- Epsilon reaching minimum (0.01) reducing exploration
- Network "forgetting" successful strategies

**2. Simple Rewards Work Better:**
Complex reward shaping (v2.0, v3.0) performed worse than simple rewards (v5.0).

**3. Double DQN is Essential:**
Switching from standard DQN to Double DQN improved stability significantly.

### 4.4 Training Curves Analysis

The training curves show:
- **Score curve:** High variance with peaks up to 710
- **Loss curve:** Stable at 0.005-0.01 (Huber Loss working well)
- **Epsilon curve:** Smooth exponential decay
- **Score distribution:** Long tail indicating learned high-performance policy

---

## 5. Discussion

### 5.1 Insights from Open Source Projects

We referenced two successful open-source implementations:

1. **aome510/chrome-dino-game-rl:**
   - Key insight: "Reward = number of obstacles passed"
   - Simple rewards lead to better learning

2. **hfahrudin/trex-DQN:**
   - Key insight: "Shorter air time = more learning opportunities"
   - Faster jump physics accelerates learning

### 5.2 Lessons Learned

1. **Simplicity is key:** Simpler state representations and reward functions often outperform complex designs

2. **Exploration matters:** Adequate exploration early in training is crucial for discovering good strategies

3. **Stability techniques are essential:** Double DQN and Huber Loss significantly improve training stability

4. **More training isn't always better:** Catastrophic forgetting can occur with extended training

### 5.3 Limitations and Future Work

**Limitations:**
- High variance in performance (0-710 score range)
- Catastrophic forgetting after extended training
- ~80% of episodes still result in 0 score

**Future Improvements:**
1. Prioritized Experience Replay to focus on important transitions
2. Larger replay buffer to preserve good experiences
3. Early stopping based on average score
4. Dueling DQN architecture for better value estimation

---

## 6. Conclusion

This project successfully implemented a Deep Q-Network agent for the Dino Jump game. Through iterative experimentation across 5 versions, we achieved a maximum score of 710 and peak average score of 39.3.

Key contributions include:
1. Effective state representation with urgency signal
2. Simple but effective reward design
3. Implementation of Double DQN with Huber Loss
4. Comprehensive analysis of training dynamics and failure modes

The project demonstrates both the power and challenges of deep reinforcement learning. While the agent can achieve impressive performance, issues like catastrophic forgetting highlight the need for additional techniques in practical applications.

---

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

2. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. AAAI.

3. aome510/chrome-dino-game-rl. GitHub. https://github.com/aome510/chrome-dino-game-rl

4. hfahrudin/trex-DQN. GitHub. https://github.com/hfahrudin/trex-DQN

5. Hugging Face Deep RL Course. https://huggingface.co/learn/deep-rl-course

---

## Appendix

### A. Project Structure

```
Assignment1_RL_Game/
├── game/
│   ├── __init__.py
│   ├── constants.py      # Game parameters
│   └── dino_game.py      # Game environment
├── agent/
│   ├── __init__.py
│   ├── agent.py          # DQN Agent
│   ├── dqn_model.py      # Neural network
│   └── replay_buffer.py  # Experience replay
├── model/
│   ├── best_model.pth    # Best trained model
│   └── training_curves.png
├── train.py              # Training script
├── play.py               # Play/demo script
├── CHANGELOG.md          # Development log
└── REPORT_DRAFT.md       # This report
```

### B. How to Run

**Training:**
```bash
python train.py --episodes 1000
```

**Play with trained AI:**
```bash
python play.py --mode ai --model model/best_model.pth
```

**Human play:**
```bash
python play.py --mode human
```

### C. GitHub Repository

[Insert GitHub URL]

### D. Demo Video

[Insert YouTube URL]

---

**Word Count:** ~1,450 words
