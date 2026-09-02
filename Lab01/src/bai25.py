"""
Bai 25. Anh xa action
Xac dinh y nghia 0,1,2,3, tao ACTION_NAMES, sinh action ngau nhien va in.
"""
import gymnasium as gym

# Theo tai lieu FrozenLake-v1 va Gymnasium
ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    print("ACTION_NAMES:", ACTION_NAMES)
    for k, v in ACTION_NAMES.items():
        print(f"  {k} -> {v}")

    action = env.action_space.sample()
    print(f"\nSinh ngau nhien: Action {action} -> {ACTION_NAMES[action]}")

    # Kiem chung bang code: thu di chuyen tu state 0 (goc trai tren)
    # State mapping 4x4: 0 1 2 3
    #                   4 5 6 7
    #                   8 9 10 11
    #                   12 13 14 15
    env2 = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    obs, info = env2.reset(seed=42)
    print(f"\nInitial state: {obs}, render:")
    print(env2.render())
    for a in [0, 1, 2, 3]:
        env2.reset(seed=42)
        obs, reward, terminated, truncated, info = env2.step(a)
        print(f"  Action {a} ({ACTION_NAMES[a]}): next_state={obs}, reward={reward}, terminated={terminated}")

    env.close()
    env2.close()


if __name__ == "__main__":
    main()
