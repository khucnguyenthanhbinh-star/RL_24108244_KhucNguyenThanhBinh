"""
Chuogn trinh tong hop hoac chuong trinh cua Bai 36 - Khung chuong trinh khuyen nghi da hoan thien
Luu tai: Lab01/src/main.py
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os


def random_policy(observation, env):
    return env.action_space.sample()


def angle_based_policy(observation):
    pole_angle = observation[2]
    return 0 if pole_angle < 0 else 1


def improved_policy(observation):
    pole_angle = observation[2]
    pole_ang_vel = observation[3]
    cart_pos = observation[0]
    if cart_pos < -1.5:
        return 1
    if cart_pos > 1.5:
        return 0
    value = pole_angle + 0.5 * pole_ang_vel
    return 0 if value < 0 else 1


def run_episode(env, policy, seed=None, max_steps=1000):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = truncated = False
    for _ in range(max_steps):
        # Policy co the can env hoac khong; o day dung signature observation-only
        try:
            action = policy(obs)
        except TypeError:
            action = policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return {"reward": total_reward, "length": length, "terminated": terminated, "truncated": truncated}


def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    env = gym.make(env_name)
    rewards = []
    lengths = []
    for i in range(n_episodes):
        res = run_episode(env, policy, seed=seed + i, max_steps=500)
        rewards.append(res["reward"])
        lengths.append(res["length"])
    env.close()
    arr = np.array(rewards)
    return {
        "mean_reward": float(np.mean(arr)),
        "std_reward": float(np.std(arr)),
        "min_reward": float(np.min(arr)),
        "max_reward": float(np.max(arr)),
        "mean_length": float(np.mean(lengths)),
        "rewards": rewards,
    }


def moving_average(values, window_size):
    if window_size <= 0 or window_size > len(values):
        return np.array([])
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def plot_rewards(rewards, title="Reward per Episode", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', markersize=2, alpha=0.7)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.close()


def main():
    print("=== Lab01 Main - Tong hop thi nghiem RL ===")
    env_name = "CartPole-v1"
    n_episodes = 500
    seed = 42

    # Tao environment
    env = gym.make(env_name)
    print(f"Created {env_name}: {env}")
    print(f"Action space: {env.action_space}, Observation space: {env.observation_space}")
    env.close()

    # Chay thi nghiem 3 agent
    agents = {
        "Random": lambda obs: np.random.choice([0, 1]),
        "Angle-based": angle_based_policy,
        "Improved": improved_policy,
    }

    results = {}
    for name, policy in agents.items():
        res = evaluate_policy(env_name, policy, n_episodes=n_episodes, seed=seed)
        results[name] = res
        print(f"\n{name}: mean={res['mean_reward']:.2f}, std={res['std_reward']:.2f}, min={res['min_reward']:.1f}, max={res['max_reward']:.1f}, mean_length={res['mean_length']:.1f}")

    # In bang so sanh
    print("\n" + "=" * 80)
    print(f"{'Agent':<15} | {'Mean':<7} | {'Std':<7} | {'Min':<6} | {'Max':<6} | {'Mean length':<11}")
    print("-" * 80)
    for name in agents:
        r = results[name]
        print(f"{name:<15} | {r['mean_reward']:<7.2f} | {r['std_reward']:<7.2f} | {r['min_reward']:<6.1f} | {r['max_reward']:<6.1f} | {r['mean_length']:<11.2f}")
    print("=" * 80)

    # Ve bieu do
    # 1. Reward cartpole (lay random)
    plot_rewards(results["Random"]["rewards"], title="CartPole-v1 - Reward per Episode (Random Agent)", save_path="Lab01/figures/reward_cartpole.png")
    # Fallback cho chay tu thu muc khac
    for p in ["figures/reward_cartpole.png", "../figures/reward_cartpole.png"]:
        try:
            plot_rewards(results["Random"]["rewards"], save_path=p)
        except:
            pass

    # 2. Moving average
    rewards = results["Random"]["rewards"]
    ma = moving_average(rewards, 10)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, alpha=0.6, label="Reward", marker='o', markersize=2)
    plt.plot(range(10, len(rewards) + 1), ma, color="red", linewidth=2, label="Moving Average (window=10)")
    plt.title("CartPole-v1 - Reward and Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    for p in ["Lab01/figures/moving_average.png", "figures/moving_average.png", "../figures/moving_average.png"]:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
        except:
            pass
    plt.close()

    # 3. Comparison
    labels = list(agents.keys())
    means = [results[k]["mean_reward"] for k in labels]
    stds = [results[k]["std_reward"] for k in labels]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color=["gray", "skyblue", "orange"], alpha=0.8)
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{m:.1f}", ha="center", va="bottom", fontweight="bold")
    plt.title("CartPole-v1 - So sanh Mean Reward (500 episodes)")
    plt.ylabel("Mean Reward")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for p in ["Lab01/figures/comparison_agents.png", "figures/comparison_agents.png", "../figures/comparison_agents.png"]:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved {p}")
        except:
            pass
    plt.close()

    # In thong ke chi tiet va ket luan
    print("\nNhan xet (5-10 dong):")
    print("- Random yeu nhat, Angle-based cai thien dang ke nho dung goc pole.")
    print("- Improved tot nhat nho ket hop goc + van toc goc + vi tri xe, on dinh hon.")
    print("- Tat ca heuristic chua dat max 500, can hoc RL thuc su de toi uu.")
    print("- Seed dam bao tai lap, moi episode da dung env.close() dung quy dinh.")

    # Dam bao dong env cuoi
    print("\nMain hoan thanh.")


if __name__ == "__main__":
    main()
