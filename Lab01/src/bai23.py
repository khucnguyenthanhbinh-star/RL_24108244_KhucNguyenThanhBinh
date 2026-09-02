"""
Bai 23. Tao FrozenLake
Tao env = gym.make("FrozenLake-v1", is_slippery=False), in observation_space, action_space, so state/so action.
"""
import gymnasium as gym


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Xac dinh bang code so state va so action
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    print(f"So state: {n_states}")
    print(f"So action: {n_actions}")
    print(f"Map size: 4x4 = 16 o (S, F, H, G)")

    env.close()


if __name__ == "__main__":
    main()
