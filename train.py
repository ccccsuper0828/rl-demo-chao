"""
Training Script for Dino Jump Game - Version 6.1 (Clean v5.0)
DQN agent with proven stable configuration

v6.1: Reverted all v6.2/v6.3 changes that caused performance degradation
- Removed dropout (harmful for simple tasks)
- Removed curriculum learning (complicated without benefit)
- Removed reward micro-adjustments (added noise)
- Back to v5.0/v6.0 proven parameters

Core features:
- Balanced epsilon decay (0.995) - enables "aha moment"
- Early stopping with warmup - prevents catastrophic forgetting
- Save best average score model - more stable than best single score
- Optional Prioritized Experience Replay

References:
- https://github.com/aome510/chrome-dino-game-rl
- https://github.com/hfahrudin/trex-DQN
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from game import DinoGame
from agent import DQNAgent


def train(
    num_episodes: int = 1000,
    max_steps: int = 10000,
    render: bool = False,
    save_freq: int = 50,
    model_dir: str = "model",
    use_per: bool = False,
    early_stop_patience: int = 100,     # v6.1: balanced patience
    early_stop_threshold: float = 0.6   # v6.1: increased from 0.4 (more sensitive)
):
    """
    Train the DQN agent with anti-forgetting mechanisms

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        render: Whether to render the game during training
        save_freq: How often to save the model
        model_dir: Directory to save models
        use_per: Use Prioritized Experience Replay
        early_stop_patience: Episodes to wait before early stopping
        early_stop_threshold: Stop if avg drops below this ratio of peak
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Initialize game and agent
    # v6.1: Clean configuration proven to work
    game = DinoGame(render=render)
    agent = DQNAgent(
        state_size=6,
        action_size=2,
        learning_rate=0.0005,       # v6.2: reduced from 0.001 to stabilize loss
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,        # v6.1: back to v5.0 (enables "aha moment")
        buffer_size=10000,          # v6.1: back to v5.0 (smaller = faster feedback loop)
        batch_size=64,              # v6.2: increased from 32 to reduce gradient variance
        target_update_freq=100,
        use_double_dqn=True,
        use_per=use_per,            # v6.0: optional PER
        soft_update=False
    )

    # Training metrics
    scores = []
    avg_scores = []
    losses = []
    epsilons = []

    # Early stopping settings
    best_score = 0
    best_avg_score = 0
    best_avg_episode = 0
    peak_avg_score = 0
    episodes_since_peak = 0
    early_stopped = False
    warmup_episodes = 300  # v6.0 fix: don't track peak until after warmup

    print("=" * 60)
    print("Starting Training - Version 6.1 (Clean)")
    print(f"Episodes: {num_episodes}")
    print(f"Device: {agent.device}")
    print(f"Buffer Size: 10000 (v5.0 - enables aha moment)")
    print(f"Epsilon Decay: 0.995 (v5.0)")
    print(f"Early Stop: warmup={warmup_episodes}, patience={early_stop_patience}")
    print(f"Saves: best_model + best_avg_model (v6.0)")
    print("=" * 60)

    for episode in range(1, num_episodes + 1):
        state = game.reset()
        total_reward = 0
        episode_loss = []

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, done, info = game.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Increase PER beta if using PER
        if use_per:
            agent.increase_per_beta(0.001)

        # Record metrics
        score = info['score']
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # Average of last 100 episodes
        avg_scores.append(avg_score)
        epsilons.append(agent.epsilon)

        if episode_loss:
            losses.append(np.mean(episode_loss))

        # Print progress
        if episode % 10 == 0:
            status = "WARMUP" if episode < warmup_episodes else f"Peak:{peak_avg_score:.1f}"
            print(f"Episode {episode:4d} | Score: {score:5d} | "
                  f"Avg Score: {avg_score:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {step:5d} | "
                  f"{status}")

        # Save best score model
        if score > best_score:
            best_score = score
            agent.save(os.path.join(model_dir, "best_model.pth"))
            print(f"  -> New best score: {best_score}")

        # Save best average score model (v6.0: prevents forgetting)
        if avg_score > best_avg_score and episode >= 100:
            best_avg_score = avg_score
            best_avg_episode = episode
            agent.save(os.path.join(model_dir, "best_avg_model.pth"))
            print(f"  -> New best avg score: {best_avg_score:.1f} at episode {episode}")

        # Track peak average for early stopping (only after warmup)
        if episode >= warmup_episodes:
            if avg_score > peak_avg_score:
                peak_avg_score = avg_score
                episodes_since_peak = 0
            else:
                episodes_since_peak += 1

        # Early stopping check (v6.0: prevent catastrophic forgetting)
        # Only check after warmup + patience episodes
        if episode >= warmup_episodes + early_stop_patience and episodes_since_peak >= early_stop_patience:
            if avg_score < peak_avg_score * early_stop_threshold:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING at episode {episode}")
                print(f"Peak avg: {peak_avg_score:.1f} (reached at ~episode {episode - episodes_since_peak})")
                print(f"Current avg: {avg_score:.1f} ({avg_score/peak_avg_score*100:.1f}% of peak)")
                print(f"No improvement for {episodes_since_peak} episodes")
                print(f"Performance dropped below {early_stop_threshold*100:.0f}% threshold")
                print(f"{'='*60}\n")
                early_stopped = True
                break

        # Periodic save
        if episode % save_freq == 0:
            agent.save(os.path.join(model_dir, f"model_ep{episode}.pth"))

    # Save final model
    agent.save(os.path.join(model_dir, "final_model.pth"))

    # Plot training curves
    plot_training_curves(scores, avg_scores, losses, epsilons, model_dir,
                        early_stopped, peak_avg_score, best_avg_episode)

    game.close()
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best score: {best_score}")
    print(f"Best avg score: {best_avg_score:.1f} (episode {best_avg_episode})")
    print(f"Peak avg score: {peak_avg_score:.1f}")
    if early_stopped:
        print("Note: Training was early stopped to prevent forgetting")
        print("Recommended: Use best_avg_model.pth for best performance")
    print("=" * 60)

    return agent, scores


def plot_training_curves(scores, avg_scores, losses, epsilons, save_dir,
                        early_stopped=False, peak_avg=0, best_avg_ep=0):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scores
    axes[0, 0].plot(scores, alpha=0.6, label='Score')
    axes[0, 0].plot(avg_scores, label='Avg Score (100 ep)')
    if peak_avg > 0:
        axes[0, 0].axhline(y=peak_avg, color='r', linestyle='--', alpha=0.5,
                          label=f'Peak Avg: {peak_avg:.1f}')
    if best_avg_ep > 0:
        axes[0, 0].axvline(x=best_avg_ep, color='g', linestyle='--', alpha=0.5,
                          label=f'Best Avg Ep: {best_avg_ep}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    title = 'Training Scores'
    if early_stopped:
        title += ' (Early Stopped)'
    axes[0, 0].set_title(title)
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    if losses:
        axes[0, 1].plot(losses)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True)

    # Epsilon
    axes[1, 0].plot(epsilons)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate')
    axes[1, 0].grid(True)

    # Score distribution
    axes[1, 1].hist(scores, bins=30, edgecolor='black')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.show()
    print(f"Training curves saved to {save_dir}/training_curves.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train DQN agent for Dino Jump v6.1')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render game during training')
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Model save frequency')
    parser.add_argument('--per', action='store_true',
                       help='Use Prioritized Experience Replay')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--patience', type=int, default=200,
                       help='Early stop patience (episodes)')

    args = parser.parse_args()

    patience = 999999 if args.no_early_stop else args.patience

    train(
        num_episodes=args.episodes,
        render=args.render,
        save_freq=args.save_freq,
        use_per=args.per,
        early_stop_patience=patience
    )
