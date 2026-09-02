"""
Bai 34. Xay dung ham evaluate_policy()
Viet evaluate_policy(env_name, policy, n_episodes=100, seed=42) tra ve mean/std/min/max/mean_length.
"""
import gymnasium as gym
import numpy as np


def run_episode(env, policy, seed=None, max_steps=1000):
    obs, info = env.reset(seed=seed)
    total = 0.0
    length = 0
    terminated = truncated = False
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        length += 1
        if terminated or truncated:
            break
    return {"reward": total, "length": length, "terminated": terminated, "truncated": truncated}


def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    """
    Danh gia policy tren env_name.
    """
    env = gym.make(env_name)
    rewards = []
    lengths = []

    for i in range(n_episodes):
        # Seed tang dan de dam bao diversity nhung van tai lap
        s = seed + i if seed is not None else None
        result = run_episode(env, policy, seed=s, max_steps=1000)
        rewards.append(result["reward"])
        lengths.append(result["length"])

    env.close()
    rewards_arr = np.array(rewards)
    lengths_arr = np.array(lengths)

    return {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "min_reward": float(np.min(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
        "mean_length": float(np.mean(lengths_arr)),
        "rewards": rewards,  # them de ve bieu do neu can
        "lengths": lengths,
    }


def random_cartpole_policy(obs):
    import random
    return random.choice([0, 1])


def main():
    result = evaluate_policy("CartPole-v1", random_cartpole_policy, n_episodes=100, seed=42)
    print(f"CartPole-v1 - Random policy (100 episodes, seed=42):")
    print(f"  Mean reward: {result['mean_reward']:.2f}")
    print(f"  Std reward: {result['std_reward']:.2f}")
    print(f"  Min: {result['min_reward']:.2f}, Max: {result['max_reward']:.2f}")
    print(f"  Mean length: {result['mean_length']:.2f}")


if __name__ == "__main__":
    main()
