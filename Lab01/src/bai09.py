"""
Bai 9. Chay 20 buoc
Viet vong lap toi da 20 timestep, in t, action, reward, dung khi terminated or truncated.
"""
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")

    for t in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"t={t:2d}, action={action}, reward={reward}")
        if terminated or truncated:
            print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated})")
            break

    env.close()


if __name__ == "__main__":
    main()
