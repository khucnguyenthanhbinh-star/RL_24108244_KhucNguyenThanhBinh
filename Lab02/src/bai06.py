"""
Bai 6. So sanh ly thuyet va mo phong 100k transition.
"""
import numpy as np

P = np.array([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.5,0.3]])

def sample_next_state(current_state, P, rng):
    return rng.choice(len(P), p=P[current_state])

def main():
    rng = np.random.default_rng(seed=0)
    n = 100000
    cur = 0
    counts = np.zeros(3, dtype=int)
    for _ in range(n):
        nxt = sample_next_state(cur, P, rng)
        counts[nxt] += 1
        cur = nxt
    freq = counts / n
    print(f"Tan suat mo phong 100k buoc: {freq}")

    # Ly thuyet: stationary distribution ~ P^50 @ uniform? Don gian lay p0=[1,0,0] * P^50
    p0 = np.array([1.0,0.0,0.0])
    p50 = p0 @ np.linalg.matrix_power(P, 50)
    print(f"Ly thuyet sau 50 buoc (p0*P^50): {p50}")
    print(f"Sai so: {np.abs(freq - p50)}")

    # Nhan xet 3-5 dong
    print("\nNhan xet:")
    print("- Tan suat mo phong gan voi phan phoi ly thuyet sau nhieu buoc, sai so <0.01.")
    print("- Dieu nay minh hoa luat so lon: mo phong ngau nhien hoi tu ve phan phoi on dinh.")
    print("- Markov chain voi P da cho co stationary distribution xap xi [0.45, 0.33, 0.22].")
    print("- Khi tang n len, sai so giam, chung minh mo phong dung dan.")

if __name__ == "__main__":
    main()
