"""
Bai 13. Chay 10 episode
Su dung random agent chay 10 episode, in bang Episode | Reward | Length.
"""
import gymnasium as gym


def random_agent(env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0
    length = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break
    return total_reward, length


def main():
    env = gym.make("CartPole-v1")
    print(f"{'Episode':<8} | {'Reward':<6} | {'Length':<6}")
    print("-" * 28)
    for ep in range(10):
        reward, length = random_agent(env)
        print(f"{ep+1:<8} | {reward:<6.1f} | {length:<6}")

    env.close()


if __name__ == "__main__":
    main()
