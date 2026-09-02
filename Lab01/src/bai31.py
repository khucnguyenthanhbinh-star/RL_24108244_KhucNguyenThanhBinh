"""
Bai 31. Policy dua tren observation
CartPole observation: [cart_pos, cart_vel, pole_angle, pole_ang_vel]. Xay dung heuristic dua tren goc pole.
"""
import gymnasium as gym
import numpy as np


def angle_based_policy(observation):
    """
    Heuristic: neu pole nghieng trai (am) thi day trai, nghieng phai (duong) thi day phai.
    observation[2] la pole angle.
    """
    pole_angle = observation[2]
    if pole_angle < 0:
        return 0  # LEFT
    else:
        return 1  # RIGHT


def random_policy(observation, env):
    return env.action_space.sample()


def evaluate_angle_policy(n_episodes=100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        total = 0.0
        for _ in range(500):
            action = angle_based_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards


def evaluate_random(n_episodes=100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        total = 0.0
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards


def main():
    n = 100
    rewards_angle = evaluate_angle_policy(n)
    rewards_random = evaluate_random(n)

    print(f"Angle-based policy ({n} episodes): mean={np.mean(rewards_angle):.2f}, std={np.std(rewards_angle):.2f}, max={np.max(rewards_angle):.2f}")
    print(f"Random policy ({n} episodes): mean={np.mean(rewards_random):.2f}, std={np.std(rewards_random):.2f}, max={np.max(rewards_random):.2f}")
    print(f" Cai tien: {np.mean(rewards_angle) - np.mean(rewards_random):.2f}")


if __name__ == "__main__":
    main()
