"""
Bai 33. Xay dung ham run_episode()
Viet ham tong quat run_episode(env, policy, seed=None, max_steps=1000) tra ve dict reward/length/terminated/truncated. Khong phu thuoc CartPole.
"""
import gymnasium as gym


def run_episode(env, policy, seed=None, max_steps=1000):
    """
    Chay 1 episode voi policy bat ky.
    Args:
        env: Gymnasium env
        policy: ham policy(observation) -> action
        seed: seed cho reset
        max_steps: so buoc toi da
    Returns:
        dict {reward, length, terminated, truncated}
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated,
        "truncated": truncated,
    }


def random_policy(obs):
    import random
    return random.choice([0, 1])


def main():
    # Test voi CartPole
    env = gym.make("CartPole-v1")
    result = run_episode(env, random_policy, seed=42, max_steps=500)
    print(f"CartPole - {result}")
    env.close()

    # Test voi FrozenLake (chung minh khong phu thuoc CartPole)
    env2 = gym.make("FrozenLake-v1", is_slippery=False)

    def frozen_random_policy(obs):
        return env2.action_space.sample()

    # Can wrap vi policy signature chi nhan obs; dung closure
    result2 = run_episode(env2, lambda obs: env2.action_space.sample(), seed=42, max_steps=100)
    print(f"FrozenLake - {result2}")
    env2.close()


if __name__ == "__main__":
    main()
