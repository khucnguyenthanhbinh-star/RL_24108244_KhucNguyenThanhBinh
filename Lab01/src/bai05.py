"""
Bai 5. Quan sat trang thai ban dau
Goi env.reset(seed=42) va in Observation, Type, Shape, Info.
Comment moi phan tu cua observation.
"""
import gymnasium as gym
import numpy as np


def main():
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)

    print(f"Observation: {observation}")
    print(f"Type: {type(observation)} - {observation.dtype if isinstance(observation, np.ndarray) else type(observation)}")
    print(f"Shape: {observation.shape if isinstance(observation, np.ndarray) else np.shape(observation)}")
    print(f"Info: {info}")

    # Comment chi tiet tung phan tu cua observation (CartPole-v1: Box(4,))
    # observation[0]: Cart Position - vi tri xe tren truc x, float32, khoang [-4.8, 4.8]
    # observation[1]: Cart Velocity - van toc xe, float32, (-inf, inf)
    # observation[2]: Pole Angle - goc pole so voi truc thang dung (rad), float32, [-0.418, 0.418] ~ [-24deg, 24deg]
    # observation[3]: Pole Angular Velocity - van toc goc cua pole, float32, (-inf, inf)
    labels = ["Cart Position (float32)", "Cart Velocity (float32)", "Pole Angle (float32)", "Pole Angular Velocity (float32)"]
    for i, (val, label) in enumerate(zip(observation, labels)):
        print(f"  observation[{i}] = {val:.6f}  # {label} - kieu {type(val).__name__}")

    env.close()


if __name__ == "__main__":
    main()
