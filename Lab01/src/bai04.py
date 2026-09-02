"""
Bai 4. Kham pha observation space
In env.observation_space va xac dinh shape, dtype, gioi han duoi/tren.
"""
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    print(f"Observation space: {env.observation_space}")

    obs_space = env.observation_space
    print(f"Shape of observation: {obs_space.shape}")
    print(f"Dtype: {obs_space.dtype}")
    print(f"Low (gioi han duoi): {obs_space.low}")
    print(f"High (gioi han tren): {obs_space.high}")

    # Giai thich gioi han: CartPole co 4 gia tri, 2 gia tri inf
    print("\nChi tiet tung chieu:")
    labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    for i, label in enumerate(labels):
        print(f"  [{i}] {label}: low={obs_space.low[i]}, high={obs_space.high[i]}")

    env.close()


if __name__ == "__main__":
    main()
