"""
Bai 29. Viet policy duoi dang ham
Tao def policy(observation) tra ve action ngau nhien, thay env.action_space.sample() bang policy(observation).
"""
import gymnasium as gym


def policy(observation):
    """
    Policy don gian: chon ngau nhien.
    observation hien chua dung nhung giu tham so de sau nay mo rong.
    """
    import random
    return random.choice([0, 1])


def policy_with_env(observation, env):
    """Ban thay the khi can env.action_space"""
    return env.action_space.sample()


def main():
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    length = 0
    for _ in range(500):
        action = policy(obs)  # thay vi env.action_space.sample()
        # Hoac: action = policy_with_env(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break

    print(f"Total reward: {total_reward}, Length: {length}")
    print("Da thay the env.action_space.sample() bang policy(observation)")

    env.close()


if __name__ == "__main__":
    main()
