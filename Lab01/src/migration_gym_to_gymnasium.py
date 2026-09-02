"""
Bai bat buoc: Chuyen code Gym cu sang Gymnasium
Code cu:
import gym
env = gym.make("CartPole-v0")
observation = env.reset()
for t in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
"""

import gymnasium as gym

# terminated co y nghia gi?
# terminated = True khi agent di den trang thai ket thuc theo ban chat bai toan,
# vi du CartPole pole do qua nguong goc hoac cart ra bien, FrozenLake roi xuong ho hoac den dich.
# Day la ket thuc tu nhien cua MDP.

# truncated co y nghia gi?
# truncated = True khi episode bi dung boi gioi han ben ngoai, chang han vuot so buoc toi da (TimeLimit, max_steps=500 cho CartPole).
# Khong phai do ban chat bai toan ma do nguoi thiet ke gioi han thoi gian.

# Vi sao khong nen dung done cua API cu?
# API cu gom done = terminated or truncated chung vao mot bien, lam mat thong tin phan biet nguyen nhan ket thuc.
# Khi hoc RL, can biet phan biet de xu ly dung: terminated thi khong boostrap, truncated thi van co the boostrap.
# Gymnasium tach rieng de nghien cuu chinh xac hon va tranh nham lan khi tinh return.


def main():
    # Code moi dung Gymnasium
    env = gym.make("CartPole-v1", render_mode=None)  # CartPole-v0 da deprecated, dung v1

    # API moi: reset tra ve (observation, info)
    observation, info = env.reset(seed=42)
    print(f"Initial observation: {observation}, info: {info}")

    for t in range(1000):
        # env.render() chi goi neu co render_mode="human" hoac "rgb_array"
        action = env.action_space.sample()

        # API moi: step tra ve 5 gia tri
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"t={t:3d}, action={action}, reward={reward}, terminated={terminated}, truncated={truncated}")

        if terminated or truncated:  # thay vi if done:
            if terminated:
                print("Episode ket thuc do terminated (ban chat bai toan)")
            if truncated:
                print("Episode ket thuc do truncated (gioi han thoi gian)")
            break

    env.close()
    print("Done - migrated to Gymnasium successfully.")


if __name__ == "__main__":
    main()
