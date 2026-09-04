"""
Bai 16. FrozenLake 4x4 is_slippery=True basic info.
"""
import gymnasium as gym

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    print(f"Number of states: {env.observation_space.n}")
    print(f"Number of actions: {env.action_space.n}")
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    env.close()

if __name__ == "__main__":
    main()
