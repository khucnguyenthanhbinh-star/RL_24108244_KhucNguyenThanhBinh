"""
Bai 2. Tao CartPole
Tao moi truong CartPole-v1 va in doi tuong env, sau do env.close()
"""
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    print(env)
    print(f"Env: {env}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
