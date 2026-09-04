"""
Bai 12. MDP 2 state, 2 action.
P[state][action] = [(prob, next_state, reward, terminated)]
"""
def create_mdp():
    P = {
        0: {
            0: [(0.7, 0, 1, False), (0.3, 1, 0, False)],
            1: [(1.0, 1, 2, False)],
        },
        1: {
            0: [(0.5, 0, 0, False), (0.5, 1, 1, False)],
            1: [(0.8, 0, 5, True), (0.2, 1, 0, False)],
        },
    }
    return P

def main():
    P = create_mdp()
    for s in sorted(P):
        for a in sorted(P[s]):
            print(f"P[{s}][{a}] = {P[s][a]}")

if __name__ == "__main__":
    main()
