"""
Bai 9. Return tu cuoi episode: G_t tu cuoi ve dau.
"""
def discounted_returns(rewards, gamma):
    n = len(rewards)
    G = [0.0]*n
    G[-1] = rewards[-1]
    for t in range(n-2, -1, -1):
        G[t] = rewards[t] + gamma * G[t+1]
    return G

def main():
    rewards = [0,0,0,1]
    for gamma in [0.9,0.99,1.0]:
        print(f"gamma={gamma}: {discounted_returns(rewards, gamma)}  # G0..G3")
    # Vi du chi tiet gamma 0.9: G3=1, G2=0+0.9*1=0.9, G1=0+0.9*0.9=0.81, G0=0.729

if __name__ == "__main__":
    main()
