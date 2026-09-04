"""
Bai 20. Deterministic vs stochastic FrozenLake state0 RIGHT.
"""
import gymnasium as gym

def show(env, state, action, is_slippery):
    P = env.unwrapped.P
    print(f"is_slippery={is_slippery}, state {state}, action {action} (RIGHT)")
    for prob, nxt, reward, terminated in P[state][action]:
        print(f"  prob {prob:.3f} -> next {nxt} reward {reward} terminated {terminated}")
    print(f"  So transition: {len(P[state][action])}")

def main():
    state, action = 0, 2  # RIGHT
    env_det = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    env_sto = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    show(env_det, state, action, False)
    show(env_sto, state, action, True)
    print("\nKet luan:")
    print("- is_slippery=False chi co 1 transition xac suat 1.0, di dung huong.")
    print("- is_slippery=True co 3 transition, moi huong 1/3 (truong hop RIGHT se co LEFT/DOWN/RIGHT).")
    print("- Stochastic lam bai toan kho hon, can DP de tinh ky vong.")
    env_det.close(); env_sto.close()

if __name__ == "__main__":
    main()
