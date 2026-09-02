"""
Bai 20. So sanh hai seed
Thuc hien seed=42 va seed=100, moi seed chay 20 episode, tinh reward trung binh.
"""
import gymnasium as gym
import numpy as np


def run_episodes_with_seed(seed, n_episodes=20):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset(seed=seed)  # seed cho moi episode? thuong seed 1 lan dau, nhung de don gian ta seed moi episode
        # Tuy nhien de dam bao ngau nhien tu nhien, chi seed lan dau; o day ta se tao env moi moi lan hoac seed incremental
        # Chuan hon: seed lan dau, cac episode sau khong seed
        # Nhung theo yeu cau de so sanh, ta lam: moi episode reset voi seed tang dan de dam bao khac nhung van thuoc nhom seed goc
        total = 0.0
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        # Tang seed cho episode tiep theo de tranh lap
        seed += 1
    env.close()
    return rewards


def main():
    # Cach chuan hon: dung seed co dinh cho moi nhom va de env tu dong stochastic sau do
    # O day ta minh hoa bang cach set seed ban dau roi khong reset seed nua
    def evaluate_seed_group(initial_seed, n_episodes=20):
        env = gym.make("CartPole-v1")
        rewards = []
        obs, info = env.reset(seed=initial_seed)
        for ep in range(n_episodes):
            if ep != 0:
                obs, info = env.reset()  # khong seed nua, de ngau nhien tu nhien
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

    rewards_42 = evaluate_seed_group(42, 20)
    rewards_100 = evaluate_seed_group(100, 20)

    mean_42 = float(np.mean(rewards_42))
    mean_100 = float(np.mean(rewards_100))

    print(f"Seed 42: rewards = {rewards_42}")
    print(f"  Mean reward (seed 42): {mean_42:.2f}")
    print(f"Seed 100: rewards = {rewards_100}")
    print(f"  Mean reward (seed 100): {mean_100:.2f}")
    print(f"Chenh lech: {abs(mean_42 - mean_100):.2f}")


if __name__ == "__main__":
    main()
