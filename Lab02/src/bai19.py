"""
Bai 19. Kiem tra tong xac suat =1 cho moi state/action.
"""
import gymnasium as gym
import numpy as np

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    P = env.unwrapped.P
    ok = True
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            total = sum(prob for prob,_,_,_ in P[s][a])
            if not np.isclose(total, 1.0):
                print(f"Invalid at s={s}, a={a}, sum={total}")
                ok = False
    print("All transitions sum to 1.0 ?" , ok)
    env.close()

if __name__ == "__main__":
    main()
