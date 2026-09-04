"""
Bai 8. Discounted return voi gamma 0,0.5,0.9,0.99,1.0.
"""
def compute_return(rewards, gamma):
    G = 0.0
    for i, r in enumerate(rewards):
        G += (gamma ** i) * r
    return G

def main():
    rewards = [1,1,1,1,1]
    print(f"{'Gamma':>6} | {'Return':>6}")
    print("-"*16)
    for g in [0.0,0.5,0.9,0.99,1.0]:
        print(f"{g:6.2f} | {compute_return(rewards,g):6.2f}")

if __name__ == "__main__":
    main()
