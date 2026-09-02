"""
Bai 32. Cai tien heuristic
Mo rong Bai 31, dung toi thieu 2 thanh phan observation: pole angle va pole angular velocity. Muc tieu mean > random.
"""
import gymnasium as gym
import numpy as np


def improved_policy(observation):
    """
    Su dung 2 thanh phan: pole angle (obs[2]) va pole angular velocity (obs[3])
    Y tuong: neu goc + van toc goc nghieng ve mot phia thi day ve phia do de can bang.
    Hoac don gian: tong co trong = angle + 0.5 * angular_velocity
    """
    pole_angle = observation[2]
    pole_ang_vel = observation[3]
    # Trong so 0.5 co the tinh chinh; dung 0.3-0.7 deu cho ket qua tot hon random
    value = pole_angle + 0.5 * pole_ang_vel
    # Them cart position de tranh ra bien? Khong bat buoc nhung giup on dinh hon
    # Neu cart gan bien, uu tien keo ve trung tam
    cart_pos = observation[0]
    # Neu cart qua bien, override
    if cart_pos < -1.5:
        return 1  # keo phai
    if cart_pos > 1.5:
        return 0  # keo trai

    if value < 0:
        return 0
    else:
        return 1


def angle_based_policy(observation):
    return 0 if observation[2] < 0 else 1


def evaluate(policy, n_episodes=100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        total = 0.0
        for _ in range(500):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards


def main():
    n = 100
    rewards_random = evaluate(lambda obs: np.random.choice([0, 1]), n)
    # random dung env sample chuan hon
    env = gym.make("CartPole-v1")
    rewards_random2 = []
    for _ in range(n):
        obs, info = env.reset()
        total = 0
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards_random2.append(total)
    env.close()

    rewards_angle = evaluate(angle_based_policy, n)
    rewards_improved = evaluate(improved_policy, n)

    print(f"Random policy: mean={np.mean(rewards_random2):.2f}, std={np.std(rewards_random2):.2f}")
    print(f"Angle-based: mean={np.mean(rewards_angle):.2f}, std={np.std(rewards_angle):.2f}")
    print(f"Improved (angle + ang_vel): mean={np.mean(rewards_improved):.2f}, std={np.std(rewards_improved):.2f}")

    if np.mean(rewards_improved) > np.mean(rewards_random2):
        print(f" -> Dat muc tieu: improved ({np.mean(rewards_improved):.2f}) > random ({np.mean(rewards_random2):.2f})")
    else:
        print(" -> Chua dat, can tinh chinh trong so")


if __name__ == "__main__":
    main()
