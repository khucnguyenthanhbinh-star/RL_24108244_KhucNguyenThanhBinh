"""
Bai 28. So sanh deterministic va stochastic
Cung chuong trinh voi is_slippery=False va is_slippery=True, moi truong hop 500 episode, so sanh success rate, avg reward, avg length. Viet ket luan bang comment.
"""
import gymnasium as gym


def evaluate_frozenlake(is_slippery, n_episodes=500):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    successes = 0
    total_reward = 0.0
    total_length = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        length = 0
        ep_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            length += 1
            if terminated or truncated:
                if reward == 1:
                    successes += 1
                break
        total_reward += ep_reward
        total_length += length

    env.close()
    return {
        "success_rate": successes / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_length": total_length / n_episodes,
        "successes": successes,
    }


def main():
    n = 500
    det = evaluate_frozenlake(False, n)
    sto = evaluate_frozenlake(True, n)

    print(f"Deterministic (is_slippery=False) - {n} episodes:")
    print(f"  Success rate: {det['success_rate']:.4f} ({det['successes']}/{n})")
    print(f"  Average reward: {det['avg_reward']:.4f}")
    print(f"  Average episode length: {det['avg_length']:.2f}")

    print(f"\nStochastic (is_slippery=True) - {n} episodes:")
    print(f"  Success rate: {sto['success_rate']:.4f} ({sto['successes']}/{n})")
    print(f"  Average reward: {sto['avg_reward']:.4f}")
    print(f"  Average episode length: {sto['avg_length']:.2f}")

    # Ket luan:
    # - Deterministic cho success_rate cao hon dang ke vi agent di dung huong da chon.
    # - Stochastic (is_slippery=True) lam agent truot ngau nhien sang huong khac (1/3 xac suat moi huong), khien viec den Goal kho hon,
    #   success_rate thap, avg_reward thap hon, avg_length co the dai hon do lang thang.
    # - Dieu nay minh hoa tinh ngau nhien cua moi truong anh huong manh den hieu qua cua random policy.
    # - De giai stochastic FrozenLake can hoc policy toi uu (Q-learning) thay vi random.


if __name__ == "__main__":
    main()
