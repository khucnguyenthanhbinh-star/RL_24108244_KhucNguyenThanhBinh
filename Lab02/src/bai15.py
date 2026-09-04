"""
Bai 15. Stochastic uniform policy.
"""
import numpy as np

def main():
    n_states, n_actions = 2, 2
    policy = np.ones((n_states, n_actions)) / n_actions
    print("Stochastic policy (uniform):")
    print(policy)
    for s in range(n_states):
        total = policy[s].sum()
        print(f"State {s}: sum = {total:.2f} {'OK' if abs(total-1)<1e-8 else 'ERR'}")

if __name__ == "__main__":
    main()
