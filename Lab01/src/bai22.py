"""
Bai 22. Ham thi nghiem co seed
Viet experiment(seed, n_episodes) tra ve dict seed/mean/std/max/min. Thuc hien voi it nhat 5 seed.
"""
import gymnasium as gym
import numpy as np


def experiment(seed, n_episodes=50):
    """
    Chay thi nghiem voi seed co dinh.
    Returns:
        dict voi seed, mean_reward, std_reward, max_reward, min_reward
    """
    env = gym.make("CartPole-v1")
    rewards = []

    # Seed lan dau
    obs, info = env.reset(seed=seed)
    # Can seed action_space neu muon tai lap hoan toan
    env.action_space.seed(seed)

    for ep in range(n_episodes):
        if ep != 0:
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
    rewards_arr = np.array(rewards)
    return {
        "seed": seed,
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
        "min_reward": float(np.min(rewards_arr)),
    }


def main():
    seeds = [42, 100, 123, 2024, 999]
    for s in seeds:
        result = experiment(s, n_episodes=50)
        print(f"Seed {result['seed']:4d}: mean={result['mean_reward']:.2f}, std={result['std_reward']:.2f}, max={result['max_reward']:.2f}, min={result['min_reward']:.2f}")


if __name__ == "__main__":
    main()
