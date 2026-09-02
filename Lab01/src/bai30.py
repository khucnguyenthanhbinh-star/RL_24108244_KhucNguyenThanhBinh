"""
Bai 30. Policy luon chon mot action
Xay dung always_left_policy va always_right_policy, chay moi policy 100 episode, so sanh mean reward.
"""
import gymnasium as gym
import numpy as np


def always_left_policy(observation):
    return 0


def always_right_policy(observation):
    return 1


def evaluate_policy(policy, n_episodes=100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        total = 0.0
        for _ in range(500):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), rewards


def main():
    mean_left, rewards_left = evaluate_policy(always_left_policy, 100)
    mean_right, rewards_right = evaluate_policy(always_right_policy, 100)

    print(f"Always LEFT policy (0) - 100 episodes:")
    print(f"  Mean reward: {mean_left:.2f}, Min: {np.min(rewards_left):.2f}, Max: {np.max(rewards_left):.2f}")
    print(f"Always RIGHT policy (1) - 100 episodes:")
    print(f"  Mean reward: {mean_right:.2f}, Min: {np.min(rewards_right):.2f}, Max: {np.max(rewards_right):.2f}")

    better = "LEFT" if mean_left > mean_right else "RIGHT"
    print(f"\nKet luan: Policy {better} co mean reward cao hon ({max(mean_left, mean_right):.2f} vs {min(mean_left, mean_right):.2f}), nhung ca hai deu kem hon random/heuristic do CartPole can can bang.")


if __name__ == "__main__":
    main()
