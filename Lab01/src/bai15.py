"""
Bai 15. Tinh thong ke reward
Tu ket qua Bai 14, tinh mean/min/max/std bang NumPy, in 2 chu so thap phan.
"""
import gymnasium as gym
import numpy as np


def random_agent(env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


def main():
    env = gym.make("CartPole-v1")
    episode_rewards = [random_agent(env) for _ in range(100)]

    rewards_array = np.array(episode_rewards)
    mean_r = np.mean(rewards_array)
    min_r = np.min(rewards_array)
    max_r = np.max(rewards_array)
    std_r = np.std(rewards_array)

    print(f"Mean reward : {mean_r:.2f}")
    print(f"Min reward  : {min_r:.2f}")
    print(f"Max reward  : {max_r:.2f}")
    print(f"Std reward  : {std_r:.2f}")

    env.close()


if __name__ == "__main__":
    main()
