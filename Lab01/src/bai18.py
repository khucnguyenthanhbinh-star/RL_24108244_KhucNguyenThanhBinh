"""
Bai 18. Moving average
Viet ham moving_average(values, window_size), tinh MA window=10, ve dong thoi reward va MA, luu moving_average.png. Khong dung Pandas.
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os


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


def moving_average(values, window_size):
    """
    Tinh moving average thu cong, khong dung pandas.
    """
    if window_size <= 0:
        raise ValueError("window_size phai > 0")
    if window_size > len(values):
        return np.array([])
    # Dung cumsum de tinh nhanh
    cumsum = np.cumsum(np.insert(values, 0, 0))
    ma = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return ma


def main():
    env = gym.make("CartPole-v1")
    n_episodes = 100
    episode_rewards = [random_agent(env) for _ in range(n_episodes)]
    env.close()

    window_size = 10
    ma = moving_average(episode_rewards, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_episodes + 1), episode_rewards, label="Reward", alpha=0.6, marker='o', markersize=2)
    # MA bat dau tu window_size
    plt.plot(range(window_size, n_episodes + 1), ma, label=f"Moving Average (window={window_size})", linewidth=2, color='red')
    plt.title("CartPole-v1 - Reward and Moving Average (window=10)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Luu
    try:
        os.makedirs("Lab01/figures", exist_ok=True)
        plt.savefig("Lab01/figures/moving_average.png", dpi=150, bbox_inches='tight')
        print("Da luu Lab01/figures/moving_average.png")
    except:
        pass
    try:
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/moving_average.png", dpi=150, bbox_inches='tight')
    except:
        pass
    try:
        os.makedirs("../figures", exist_ok=True)
        plt.savefig("../figures/moving_average.png", dpi=150, bbox_inches='tight')
    except:
        pass

    print(f"MA cuoi cung: {ma[-1]:.2f}" if len(ma) > 0 else "Khong tinh duoc MA")
    plt.close()


if __name__ == "__main__":
    main()
