"""
Bai 19. Thu nghiem voi seed
Chay env.reset(seed=42) 10 lan trong 10 env doc lap va ghi lai initial observation, kiem tra co giong nhau khong.
"""
import gymnasium as gym
import numpy as np


def main():
    observations = []
    for i in range(10):
        env = gym.make("CartPole-v1")
        obs, info = env.reset(seed=42)
        observations.append(obs)
        print(f"Env {i+1}: obs = {obs}")
        env.close()

    # Kiem tra co giong nhau khong
    first = observations[0]
    all_equal = all(np.allclose(obs, first) for obs in observations)
    print(f"\nTat ca observation co giong nhau khong? {all_equal}")

    # Ket luan 2-3 dong comment:
    # Khi dung cung seed=42 cho env.reset(), moi truong khoi tao cung trang thai ban dau.
    # Dieu nay giup thi nghiem tai lap duoc (reproducible). Neu seed khac nhau, observation se khac.
    # Viec seed hoa quan trong de so sanh cong bang giua cac policy/agent.
    if all_equal:
        print("Ket luan: Cung seed -> cung initial observation. Thi nghiem co the tai lap.")
    else:
        print("Ket luan: Khac nhau -> co van de ve seed hoac stochasticity chua kiem soat.")


if __name__ == "__main__":
    main()
