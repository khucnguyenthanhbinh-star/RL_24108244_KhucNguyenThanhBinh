"""
Bai 17. In transition model state 0.
"""
import gymnasium as gym

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    P = env.unwrapped.P
    state = 0
    for action in range(env.action_space.n):
        print(f"\nAction {action}:")
        for prob, nxt, reward, terminated in P[state][action]:
            print(f"  Probability {prob:.3f}, Next state {nxt}, Reward {reward}, Terminated {terminated}")
    env.close()

if __name__ == "__main__":
    main()
