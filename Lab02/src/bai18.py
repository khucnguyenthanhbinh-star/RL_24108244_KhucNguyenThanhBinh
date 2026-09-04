"""
Bai 18. Ham describe_state.
"""
import gymnasium as gym

def describe_state(env, state):
    P = env.unwrapped.P
    print(f"\nState {state}:")
    for action in range(env.action_space.n):
        print(f" Action {action}:")
        for prob, nxt, reward, terminated in P[state][action]:
            print(f"   -> prob {prob:.2f} to {nxt} reward {reward} terminated {terminated}")

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    for s in [0,1,14]:
        describe_state(env, s)
    env.close()

if __name__ == "__main__":
    main()
