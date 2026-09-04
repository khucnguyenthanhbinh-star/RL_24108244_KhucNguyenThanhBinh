"""
Bai 4. Distribution sau nhieu buoc: p0 * P^n.
"""
import numpy as np

P = np.array([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.5,0.3]])

def state_distribution(p0, P, n_steps):
    """Tinh p_n = p0 * P^n"""
    if n_steps == 0:
        return p0.copy()
    Pn = np.linalg.matrix_power(P, n_steps)
    return p0 @ Pn

def main():
    p0 = np.array([1.0, 0.0, 0.0])
    for t in [1,2,5,10,50]:
        pn = state_distribution(p0, P, t)
        print(f"t={t:2d}: {pn}  (tong {pn.sum():.4f})")
    # Ve bieu do
    import matplotlib.pyplot as plt
    import os
    steps = [0,1,2,5,10,50]
    dists = np.array([state_distribution(p0,P,s) for s in steps])
    plt.figure(figsize=(10,6))
    for i, label in enumerate(["Sunny","Cloudy","Rainy"]):
        plt.plot(steps, dists[:,i], marker='o', label=label)
    plt.title("Markov Distribution theo buoc")
    plt.xlabel("n_steps"); plt.ylabel("Probability"); plt.legend(); plt.grid(True, alpha=0.3)
    os.makedirs("Lab02/figures", exist_ok=True)
    plt.savefig("Lab02/figures/markov_distribution.png", dpi=150, bbox_inches='tight')
    # fallback
    for p in ["Lab02/figures/markov_distribution.png","figures/markov_distribution.png","../figures/markov_distribution.png"]:
        try:
            import os as _os; _os.makedirs(_os.path.dirname(p), exist_ok=True); plt.savefig(p, dpi=150, bbox_inches='tight')
        except: pass
    print("Da luu markov_distribution.png")
    plt.close()

if __name__ == "__main__":
    main()
