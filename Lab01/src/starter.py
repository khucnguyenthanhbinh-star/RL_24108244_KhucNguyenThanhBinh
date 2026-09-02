"""
Chuong trinh khoi dong (starter) - khong duoc chi chay file nay roi nop
Minh hoa vong lap Agent-Environment co ban voi Gymnasium
Luu tai: Lab01/src/starter.py
"""
import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
print("Initial observation:", observation)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

for t in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(
        f"step={t:3d}, "
        f"action={action}, "
        f"reward={reward}, "
        f"terminated={terminated}, "
        f"truncated={truncated}"
    )
    if terminated or truncated:
        print("Episode ended.")
        break

env.close()
