"""
Bai 8. Viet ham run_one_step()
Ham tra ve observation, reward, terminated, truncated, info. Kiem thu voi it nhat 5 action.
"""
import gymnasium as gym


def run_one_step(env, action):
    """
    Thuc hien mot buoc tuong tac.
    Args:
        env: Gymnasium environment
        action: action muon thuc hien
    Returns:
        observation, reward, terminated, truncated, info
    """
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated, info


def main():
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    print(f"Initial state: {obs}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = run_one_step(env, action)
        print(f"Test {i+1}: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}, obs={obs}")
        if terminated or truncated:
            print("  -> Episode ended, reseting...")
            obs, info = env.reset()
        else:
            print(f"  -> Next obs: {obs}")

    env.close()


if __name__ == "__main__":
    main()
