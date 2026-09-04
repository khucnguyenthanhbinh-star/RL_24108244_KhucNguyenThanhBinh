"""
Bai 7. Undiscounted return gamma=1.0.
"""
def compute_return(rewards, gamma):
    G = 0.0
    for i, r in enumerate(rewards):
        G += (gamma ** i) * r
    return G

def main():
    rewards = [1,1,1,1,1]
    print(f"gamma=1.0, rewards {rewards} -> Return = {compute_return(rewards,1.0)}")

if __name__ == "__main__":
    main()
