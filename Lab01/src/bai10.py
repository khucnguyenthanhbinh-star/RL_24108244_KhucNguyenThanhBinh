"""
Bai 10. Tong reward
Mo rong Bai 9, tinh total_reward va in Episode length, Total reward.
"""
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    episode_length = 0

    for t in range(500):  # CartPole gioi han 500 buoc
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        print(f"t={t:3d}, action={action}, reward={reward}, total={total_reward}")
        if terminated or truncated:
            print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated})")
            break

    print(f"\nEpisode length: {episode_length}")
    print(f"Total reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
