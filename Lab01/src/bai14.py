"""
Bai 14. Chay 100 episode
Chay random agent 100 episode, luu reward vao episode_rewards.
"""
import gymnasium as gym


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
    episode_rewards = []

    for ep in range(100):
        reward = random_agent(env)
        episode_rewards.append(reward)
        # Khong in tung timestep theo yeu cau

    print(f"Da chay 100 episode. Luu {len(episode_rewards)} reward.")
    print(f"5 reward dau: {episode_rewards[:5]}")
    print(f"5 reward cuoi: {episode_rewards[-5:]}")

    env.close()


if __name__ == "__main__":
    main()
