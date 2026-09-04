"""
Bai 11. Reward som vs tre: A=[5,0,0,0,0], B=[0,0,0,0,10], tim gamma B>A.
"""
import numpy as np

def compute_return(rewards, gamma):
    return sum((gamma**i)*r for i,r in enumerate(rewards))

def main():
    A = [5,0,0,0,0]
    B = [0,0,0,0,10]
    print(f"{'gamma':>6} | {'A':>6} | {'B':>6} | {'B>A?'}")
    for g in [0.0,0.5,0.7,0.8,0.9,0.95,0.99,1.0]:
        a=compute_return(A,g); b=compute_return(B,g)
        print(f"{g:6.2f} | {a:6.2f} | {b:6.2f} | {b>a}")
    # Tim nguong gamma* sao B>A: giai 5 < 10*gamma^4 => gamma > (0.5)^{1/4}
    thresh = (0.5)**0.25
    print(f"Nguong ly thuyet: gamma > {thresh:.4f} thi B>A")
    # Dung code tim
    gammas = np.linspace(0,1,1001)
    cross = [g for g in gammas if compute_return(B,g) > compute_return(A,g)]
    if cross:
        print(f"Tim bang code: B>A khi gamma >= {min(cross):.3f}")

if __name__ == "__main__":
    main()
