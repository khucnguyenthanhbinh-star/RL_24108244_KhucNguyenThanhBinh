"""
Bai 27. Reward trong FrozenLake
Chay it nhat 100 episode bang random policy, dem success/failure, tinh success_rate.
"""
import gymnasium as gym


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    total_episodes = 100
    success = 0
    failure = 0

    for ep in range(total_episodes):
        obs, info = env.reset()
        done = False
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if reward == 1:
                    success += 1
                else:
                    failure += 1
                break
        else:
            failure += 1

    success_rate = success / total_episodes
    print(f"Total episodes: {total_episodes}")
    print(f"Success: {success}")
    print(f"Failure: {failure}")
    print(f"Success rate: {success_rate:.4f} ({success_rate*100:.2f}%)")

    env.close()


if __name__ == "__main__":
    main()
