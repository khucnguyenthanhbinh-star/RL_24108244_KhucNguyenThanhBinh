"""
Bai 14. Deterministic policy: policy[s]=a
"""
import numpy as np

def print_policy(policy):
    for s, a in enumerate(policy):
        print(f"State {s}: Action {a}")

def main():
    # MDP 2 state, 2 action
    policy = np.array([0, 1])  # s0->a0, s1->a1
    print("Deterministic policy:", policy)
    print_policy(policy)

if __name__ == "__main__":
    main()
