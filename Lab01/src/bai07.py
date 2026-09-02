"""
Bai 7. Mot buoc tuong tac
Tao CartPole va chi thuc hien dung 1 action. In State before/after, Reward, Terminated, Truncated, Info.
"""
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    print(f"State before action: {observation}")

    action = env.action_space.sample()
    print(f"Action: {action} ({'LEFT (0)' if action == 0 else 'RIGHT (1)'})")

    next_observation, reward, terminated, truncated, info = env.step(action)

    print(f"State after action: {next_observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    print(f"Done (terminated or truncated): {terminated or truncated}")

    env.close()


if __name__ == "__main__":
    main()
