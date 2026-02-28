"""
Play Script - Version 5.0
Play the game with trained AI or human control
"""

import os
import argparse
import pygame

from game import DinoGame
from agent import DQNAgent


def play_with_ai(model_path: str, num_games: int = 5):
    """
    Play game with trained AI agent

    Args:
        model_path: Path to trained model
        num_games: Number of games to play
    """
    # Initialize game and agent
    # v7.0: 6-dimensional state (added speed + obstacle height)
    game = DinoGame(render=True)
    agent = DQNAgent(state_size=6, action_size=2)

    # Load trained model
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0  # No exploration during play
    else:
        print(f"Model not found: {model_path}")
        print("Playing with random actions...")

    print("\n" + "=" * 50)
    print("AI Playing Dino Jump")
    print("Press ESC to quit, R to restart")
    print("=" * 50 + "\n")

    scores = []

    for game_num in range(1, num_games + 1):
        state = game.reset()
        done = False

        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game.close()
                        return

            # AI selects action
            action = agent.select_action(state, training=False)

            # Execute action
            state, reward, done, info = game.step(action)

        scores.append(info['score'])
        print(f"Game {game_num}: Score = {info['score']}, "
              f"Obstacles passed = {info['obstacles_passed']}")

        # Wait a bit before next game
        pygame.time.wait(1000)

    print(f"\nAverage score: {sum(scores) / len(scores):.1f}")
    print(f"Best score: {max(scores)}")

    game.close()


def play_human():
    """Play game with human control"""
    game = DinoGame(render=True)

    print("\n" + "=" * 50)
    print("Human Playing Dino Jump")
    print("Controls:")
    print("  SPACE/UP/W = Jump")
    print("  DOWN/S = Duck")
    print("  R = Restart")
    print("  ESC = Quit")
    print("=" * 50 + "\n")

    running = True
    while running:
        running, action = game.handle_human_input()
        if running and action is not None:
            state, reward, done, info = game.step(action)
            if done:
                print(f"Game Over! Score: {info['score']}")

    game.close()


def compare_ai_human(model_path: str):
    """
    Interactive mode: Watch AI play, then try yourself

    Args:
        model_path: Path to trained model
    """
    print("\n" + "=" * 50)
    print("AI vs Human Comparison Mode")
    print("=" * 50)

    # AI plays first
    print("\n--- AI Playing ---")
    game = DinoGame(render=True)
    agent = DQNAgent(state_size=6, action_size=2)  # v7.0: 6-dimensional state

    if os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0

    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                return

        action = agent.select_action(state, training=False)
        state, reward, done, info = game.step(action)

    ai_score = info['score']
    print(f"AI Score: {ai_score}")

    pygame.time.wait(2000)

    # Human plays
    print("\n--- Your Turn! ---")
    state = game.reset()
    done = False

    while not done:
        running, action = game.handle_human_input()
        if not running:
            break
        if action is not None:
            state, reward, done, info = game.step(action)

    human_score = info['score']
    print(f"Your Score: {human_score}")

    # Compare
    print("\n--- Results ---")
    print(f"AI Score: {ai_score}")
    print(f"Your Score: {human_score}")
    if human_score > ai_score:
        print("You WIN!")
    elif human_score < ai_score:
        print("AI WINS!")
    else:
        print("It's a TIE!")

    game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Dino Jump')
    parser.add_argument('--mode', choices=['ai', 'human', 'compare'],
                       default='ai', help='Play mode')
    parser.add_argument('--model', type=str, default='model/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--games', type=int, default=5,
                       help='Number of games for AI mode')

    args = parser.parse_args()

    if args.mode == 'ai':
        play_with_ai(args.model, args.games)
    elif args.mode == 'human':
        play_human()
    elif args.mode == 'compare':
        compare_ai_human(args.model)
