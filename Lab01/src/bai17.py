"""
Bai 17. Ve reward theo episode
Dung Matplotlib ve truc X: episode, Y: total reward, co title/xlabel/ylabel/grid, luu vao figures/reward_cartpole.png
"""
import gymnasium as gym
import matplotlib.pyplot as plt
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


def main():
    env = gym.make("CartPole-v1")
    n_episodes = 100
    episode_rewards = [random_agent(env) for _ in range(n_episodes)]
    env.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_episodes + 1), episode_rewards, marker='o', linestyle='-', alpha=0.7, markersize=3)
    plt.title("CartPole-v1 - Reward per Episode (Random Agent)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    # Luu dung duong dan yeu cau
    save_path = "figures/reward_cartpole.png"
    # Neu chay tu Lab01/src/ thi can ../figures
    if not os.path.exists("figures"):
        os.makedirs("../figures", exist_ok=True)
        save_path = "../figures/reward_cartpole.png"
        # fallback: thu ca hai vi tri
        plt.savefig("figures/reward_cartpole.png", dpi=150, bbox_inches='tight')
        plt.savefig("../figures/reward_cartpole.png", dpi=150, bbox_inches='tight')
    else:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Dam bao luu dung cho Lab01/figures khi chay tu goc repo
    try:
        plt.savefig("Lab01/figures/reward_cartpole.png", dpi=150, bbox_inches='tight')
    except:
        pass
    try:
        plt.savefig("../figures/reward_cartpole.png", dpi=150, bbox_inches='tight')
    except:
        pass

    print(f"Da luu bieu do vao {save_path}")
    print(f"Mean reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    # plt.show()  # comment neu chay headless
    plt.close()


if __name__ == "__main__":
    main()
