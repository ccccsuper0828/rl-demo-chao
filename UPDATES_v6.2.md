# Version 6.2 Updates - Curriculum Learning & Optimization

**Date:** 2026-02-09

## Summary

Version 6.2 introduces three major optimizations focused on training stability and learning efficiency:

1. **Dropout Regularization** - Prevents overfitting
2. **Curriculum Learning** - Progressive difficulty training
3. **Optimized Reward Function** - Better learning signals

---

## 1. Dropout Regularization (agent/dqn_model.py)

### Changes
- Added `nn.Dropout(0.2)` after each hidden layer
- Prevents overfitting on training data
- Improves generalization to unseen game states

### Code Changes
```python
# Before (v6.1)
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

# After (v6.2)
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout1(x)      # NEW
    x = F.relu(self.fc2(x))
    x = self.dropout2(x)      # NEW
    return self.fc3(x)
```

### Expected Benefits
- **Reduced overfitting**: Network won't memorize specific game patterns
- **Better generalization**: Performs more consistently across different game states
- **Stable training**: Less variance in performance

---

## 2. Curriculum Learning (game/dino_game.py)

### Concept
Train the agent progressively from easy to hard difficulty, similar to how humans learn:
- **Stage 1 (Episodes 0-300)**: Easy - Learn basic jumping
- **Stage 2 (Episodes 300-700)**: Medium - Improve timing
- **Stage 3 (Episodes 700+)**: Hard - Master the game

### Difficulty Parameters

| Parameter | Easy | Medium | Hard |
|-----------|------|--------|------|
| Initial Speed | 3.5 | 5.0 | 5.0 |
| Max Speed | 6.0 | 8.5 | 10.0 |
| Gap Min | 500 | 450 | 400 |
| Gap Max | 700 | 650 | 600 |
| Speed Increment | 0.0003 | 0.0004 | 0.0005 |

### Implementation
```python
class DinoGame:
    def __init__(self, render=True, curriculum_learning=True):
        self.curriculum_learning = curriculum_learning
        self.current_episode = 0
        self.difficulty_level = 0  # 0=easy, 1=medium, 2=hard

    def set_episode(self, episode: int):
        """Called at start of each episode"""
        self.current_episode = episode
        self._update_difficulty()

    def _update_difficulty(self):
        if self.current_episode < 300:
            self.difficulty_level = 0  # Easy
        elif self.current_episode < 700:
            self.difficulty_level = 1  # Medium
        else:
            self.difficulty_level = 2  # Hard
```

### Expected Benefits
- **Faster initial learning**: Agent learns basics on easy difficulty
- **More stable progression**: Gradual increase prevents frustration
- **Better final performance**: Solid foundation leads to better mastery
- **Reduced catastrophic forgetting**: Progressive learning is more robust

---

## 3. Optimized Reward Function (game/dino_game.py)

### Philosophy
Keep the simple base rewards, but add micro-signals for better guidance.

### Reward Structure

| Event | Reward | Purpose |
|-------|--------|---------|
| Pass obstacle | +1.0 | Main goal (unchanged) |
| Die | -1.0 | Penalty (unchanged) |
| Survive (base) | +0.01 | Keep trying (unchanged) |
| Jump when far (>200px) | -0.005 | **NEW**: Discourage wasteful jumps |
| On ground when close (50-150px) | +0.005 | **NEW**: Encourage ready position |

### Code
```python
def _calculate_reward(self, collision, passed):
    if collision:
        return -1.0
    if passed > 0:
        return 1.0

    reward = 0.01  # Base survival

    # NEW: Micro-penalty for inefficient jumping
    if self.dino.is_jumping and dist > 200:
        reward -= 0.005

    # NEW: Micro-bonus for ready position
    if not self.dino.is_jumping and 50 < dist < 150:
        reward += 0.005

    return reward
```

### Expected Benefits
- **More efficient jumps**: Agent learns not to jump unnecessarily
- **Better timing**: Encouraged to wait in ready position
- **Clearer signals**: Small but consistent feedback
- **No disruption**: Base rewards still dominate (1.0 vs 0.005)

---

## Training Script Updates (train.py)

### New Features
1. Automatic curriculum progression
2. Difficulty display in training output
3. Command-line flag to disable curriculum

### Usage

```bash
# Standard training with curriculum learning (recommended)
python train.py --episodes 1000

# Disable curriculum learning (use fixed hard difficulty)
python train.py --episodes 1000 --no-curriculum

# With PER and custom patience
python train.py --episodes 1500 --per --patience 150

# Watch training with rendering
python train.py --episodes 500 --render
```

### Training Output
```
Episode  300 | Score:   120 | Avg:   18.3 | ε: 0.612 | Diff: Easy   | Peak:18.3
Episode  600 | Score:   240 | Avg:   32.5 | ε: 0.303 | Diff: Medium | Peak:32.5
Episode  900 | Score:   450 | Avg:   41.2 | ε: 0.122 | Diff: Hard   | Peak:41.2
```

---

## Expected Training Results

### Hypothesis

| Metric | v6.1 | v6.2 (Expected) | Improvement |
|--------|------|-----------------|-------------|
| Learning Speed | Medium | **Fast** | 30-40% faster |
| Peak Avg Score | 39.3 | **45-50** | +15-25% |
| Training Stability | Good | **Excellent** | Less variance |
| Catastrophic Forgetting | Occasional | **Rare** | Better retention |
| 0-score episodes | ~70% | **~50%** | More successes |

### Why These Improvements?

1. **Curriculum Learning**
   - Faster initial learning on easy mode
   - Builds solid foundation before facing challenges
   - Reduces early frustration and random exploration

2. **Dropout**
   - Prevents network from memorizing specific patterns
   - Forces learning of robust, generalizable strategies
   - Reduces performance variance

3. **Optimized Rewards**
   - Micro-signals guide without overwhelming
   - Encourages efficient, well-timed jumps
   - Complements main rewards rather than replacing them

---

## Comparison with v6.1

### What's Kept from v6.1
✅ Buffer size: 10,000 (enables "aha moment")
✅ Epsilon decay: 0.995 (balanced exploration)
✅ Early stopping with warmup
✅ Best average model saving
✅ Double DQN
✅ Huber Loss

### What's New in v6.2
🆕 Curriculum learning (3-stage progression)
🆕 Dropout layers (0.2 rate)
🆕 Optimized reward function with micro-signals
🆕 Episode-aware difficulty adjustment

---

## Testing Recommendations

### Quick Test (500 episodes)
```bash
python train.py --episodes 500
```
Expected: avg score ~25-30 by episode 500

### Standard Training (1000 episodes)
```bash
python train.py --episodes 1000
```
Expected: avg score ~40-45, best score 500+

### Long Training (1500 episodes)
```bash
python train.py --episodes 1500 --patience 150
```
Expected: avg score peaks ~episode 1000, early stop ~1150

### Comparison Test (no curriculum)
```bash
python train.py --episodes 1000 --no-curriculum
```
Expected: Similar to v6.1 performance (slower learning)

---

## Files Modified

1. **agent/dqn_model.py**
   - Added dropout layers
   - Added dropout_rate parameter

2. **game/dino_game.py**
   - Added curriculum_learning parameter
   - Added set_episode() method
   - Added _update_difficulty() method
   - Added _get_difficulty_params() method
   - Optimized _calculate_reward() method
   - Updated _spawn_obstacle() to use dynamic gaps
   - Updated speed calculation to use dynamic parameters

3. **train.py**
   - Added curriculum_learning parameter
   - Added game.set_episode() call in training loop
   - Updated training output to show difficulty
   - Added --no-curriculum command-line flag
   - Updated version string to 6.2

---

## Troubleshooting

### If training is too slow:
- Reduce episodes in easy stage: modify `if self.current_episode < 200:` (instead of 300)
- Skip to medium: start from episode 300

### If performance is worse:
- Disable curriculum: `--no-curriculum`
- Reduce dropout: edit `dropout_rate=0.1` in dqn_model.py

### If early stopping triggers too early:
- Increase patience: `--patience 200`
- Disable early stop: `--no-early-stop`

---

## Next Steps

After verifying v6.2 results:

1. **If successful** (avg score >45):
   - Consider adding Dueling DQN
   - Try PER with curriculum
   - Experiment with 4-stage curriculum

2. **If marginal** (avg score 35-40):
   - Tune curriculum transition points
   - Adjust reward micro-signals
   - Try different dropout rates

3. **If worse** (avg score <35):
   - Revert to v6.1
   - Analyze which component caused issues
   - Test components individually

---

## References

- Curriculum Learning: Bengio et al. (2009) "Curriculum Learning"
- Dropout: Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- Progressive Training: OpenAI (2019) "Emergent Tool Use from Multi-Agent Interaction"
