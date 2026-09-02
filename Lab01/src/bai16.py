"""
Bai 16. Episode tot nhat
Tim episode co reward lon nhat, reward va do dai tuong ung. Khong duyet lai moi truong.
"""
import gymnasium as gym
import numpy as np


def random_agent_with_length(env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0
    length = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return total_reward, length


def main():
    env = gym.make("CartPole-v1")
    n_episodes = 100
    rewards = []
    lengths = []

    for _ in range(n_episodes):
        r, l = random_agent_with_length(env)
        rewards.append(r)
        lengths.append(l)

    rewards_arr = np.array(rewards)
    best_idx = int(np.argmax(rewards_arr))

    print(f"Best episode index: {best_idx + 1} (1-based)")
    print(f"Reward tuong ung: {rewards[best_idx]:.2f}")
    print(f"Do dai episode tuong ung: {lengths[best_idx]}")

    # Them thong tin phu
    print(f"\nWorst episode: index {int(np.argmin(rewards_arr))+1}, reward {np.min(rewards_arr):.2f}, length {lengths[int(np.argmin(rewards_arr))]}")

    env.close()


if __name__ == "__main__":
    main()
