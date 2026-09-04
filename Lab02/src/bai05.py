"""
Bai 5. Mo phong Markov chain: sample_next_state dung rng.choice.
"""
import numpy as np

P = np.array([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.5,0.3]])
states = ["Sunny","Cloudy","Rainy"]

def sample_next_state(current_state, P, rng):
    """Lay mau state ke tiep theo phan phoi P[current_state]"""
    return rng.choice(len(P), p=P[current_state])

def main():
    rng = np.random.default_rng(seed=42)
    current = 0  # Sunny
    seq = [current]
    for _ in range(30):
        nxt = sample_next_state(current, P, rng)
        seq.append(nxt)
        current = nxt
    print("Chuoi 31 state (30 transition):")
    print(seq)
    print([states[s] for s in seq])

if __name__ == "__main__":
    main()
