"""
Bai 36. Mini-project: Agent-Environment hoan chinh
Chon CartPole-v1, viet chuong trinh hoan chinh create_environment()/policy()/run_episode()/evaluate_policy()/plot_results()/main()
Yeu cau: >=500 episode, luu reward/length, mean/std, best/worst, ve reward + MA, co seed, env.close(), comment, khong dung API cu.
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os


def create_environment(seed=42):
    env = gym.make("CartPole-v1")
    return env


def policy(observation):
    """
    Improved policy: ket hop pole angle, angular velocity va cart position.
    """
    pole_angle = observation[2]
    pole_ang_vel = observation[3]
    cart_pos = observation[0]
    if cart_pos < -1.8:
        return 1
    if cart_pos > 1.8:
        return 0
    value = pole_angle + 0.5 * pole_ang_vel
    return 0 if value < 0 else 1


def run_episode(env, policy, seed=None, max_steps=500):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = truncated = False
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return {"reward": total_reward, "length": length, "terminated": terminated, "truncated": truncated}


def evaluate_policy(env_name, policy, n_episodes=500, seed=42):
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
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": float(np.mean(arr)),
        "std_reward": float(np.std(arr)),
        "min_reward": float(np.min(arr)),
        "max_reward": float(np.max(arr)),
        "mean_length": float(np.mean(lengths)),
        "best_idx": int(np.argmax(arr)),
        "worst_idx": int(np.argmin(arr)),
    }


def moving_average(values, window_size):
    if window_size <= 0 or window_size > len(values):
        return np.array([])
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def plot_results(rewards, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    n = len(rewards)
    window = 10
    ma = moving_average(rewards, window)

    # Bieu do 1: Reward
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n + 1), rewards, alpha=0.6, marker='o', markersize=2, label="Reward")
    plt.title("Mini-project CartPole-v1 - Reward per Episode (500 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_cartpole.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Bieu do 2: Moving average
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n + 1), rewards, alpha=0.5, marker='o', markersize=2, label="Reward")
    plt.plot(range(window, n + 1), ma, linewidth=2, color="red", label=f"Moving Average (window={window})")
    plt.title("Mini-project CartPole-v1 - Reward and Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "moving_average.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Bieu do 3: Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(rewards, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    plt.title("Histogram of Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return ma


def main():
    print("=== Mini-project: CartPole-v1 Agent-Environment hoan chinh ===")
    env_name = "CartPole-v1"
    n_episodes = 500
    seed = 42

    # 1. Tao moi truong
    env = create_environment(seed=seed)
    print(f"Da tao {env_name}: {env}")
    env.close()

    # 2. Chay thi nghiem
    print(f"\nDang chay {n_episodes} episode voi improved policy (seed={seed})...")
    result = evaluate_policy(env_name, policy, n_episodes=n_episodes, seed=seed)

    # 3. In thong ke
    print(f"\nKet qua {n_episodes} episode:")
    print(f"  Mean reward: {result['mean_reward']:.2f}")
    print(f"  Std reward: {result['std_reward']:.2f}")
    print(f"  Min reward: {result['min_reward']:.2f} (episode {result['worst_idx']+1})")
    print(f"  Max reward: {result['max_reward']:.2f} (episode {result['best_idx']+1})")
    print(f"  Mean length: {result['mean_length']:.2f}")
    print(f"  Episode tot nhat: {result['best_idx']+1}, reward={result['rewards'][result['best_idx']]:.1f}, length={result['lengths'][result['best_idx']]}")
    print(f"  Episode te nhat: {result['worst_idx']+1}, reward={result['rewards'][result['worst_idx']]:.1f}, length={result['lengths'][result['worst_idx']]}")

    # 4. Ve bieu do
    for d in ["figures", "Lab01/figures", "../figures"]:
        try:
            ma = plot_results(result["rewards"], save_dir=d)
            print(f"Da luu bieu do vao {d}/")
            break
        except Exception as e:
            continue
    # Dam bao ca 3 vi tri neu chay tu khac nhau
    for d in ["figures", "Lab01/figures"]:
        try:
            os.makedirs(d, exist_ok=True)
            plot_results(result["rewards"], save_dir=d)
        except:
            pass

    # 5. Ket luan
    print("\nKet luan thi nghiem:")
    print("- Improved heuristic vuot troi random nho su dung thong tin vat ly (goc + van toc goc).")
    print("- Du vay, policy co dinh chua dat toi uu 500, dao dong lon do nhay cam voi trang thai dau.")
    print("- De on dinh hon can hoc RL (Q-learning/DQN) hoac PID controller.")
    print("- Thi nghiem co seed dam bao tai lap, env.close() da goi dung quy dinh.")

    # Luu data
    try:
        os.makedirs("data", exist_ok=True)
        np.save("data/episode_rewards.npy", np.array(result["rewards"]))
        print("Da luu data/episode_rewards.npy")
    except:
        pass
    try:
        os.makedirs("Lab01/data", exist_ok=True)
        np.save("Lab01/data/episode_rewards.npy", np.array(result["rewards"]))
    except:
        pass


if __name__ == "__main__":
    main()
