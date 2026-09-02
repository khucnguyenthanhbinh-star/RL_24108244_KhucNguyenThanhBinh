"""
Bai 11. Viet random agent
Ham random_agent(env, max_steps=500) reset, chon action ngau nhien den khi ket thuc, tra ve total_reward, episode_length.
"""
import gymnasium as gym


def random_agent(env, max_steps=500):
    """
    Random agent chay 1 episode.
    Returns:
        total_reward, episode_length
    """
    obs, info = env.reset()
    total_reward = 0.0
    episode_length = 0

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        if terminated or truncated:
            break

    return total_reward, episode_length


def main():
    env = gym.make("CartPole-v1")
    # Test voi seed de tai lap cho demo
    obs, info = env.reset(seed=42)
    # Chay 1 episode bang random_agent (ham se reset lai ben trong)
    total_reward, length = random_agent(env, max_steps=500)
    print(f"Total reward: {total_reward}")
    print(f"Episode length: {length}")

    # Chay them 5 episode de minh hoa
    print("\nChay 5 episode:")
    for i in range(5):
        r, l = random_agent(env)
        print(f"  Episode {i+1}: reward={r}, length={l}")

    env.close()


if __name__ == "__main__":
    main()
