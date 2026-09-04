"""
Bai 10. Anh huong gamma: rewards [0,0,0,0,10], ve G0 theo gamma.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_return(rewards, gamma):
    G=0.0
    for i,r in enumerate(rewards):
        G+=(gamma**i)*r
    return G

def main():
    rewards = [0,0,0,0,10]
    gammas = np.linspace(0,1,101)
    G0s = [compute_return(rewards,g) for g in gammas]
    plt.figure(figsize=(8,5))
    plt.plot(gammas, G0s, label="G0")
    plt.title("G0 theo gamma (rewards [0,0,0,0,10])")
    plt.xlabel("gamma"); plt.ylabel("G0"); plt.grid(True, alpha=0.3); plt.legend()
    os.makedirs("Lab02/figures", exist_ok=True)
    plt.savefig("Lab02/figures/gamma_comparison.png", dpi=150, bbox_inches='tight')
    for p in ["figures/gamma_comparison.png","../figures/gamma_comparison.png"]:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True); plt.savefig(p, dpi=150, bbox_inches='tight')
        except: pass
    print("Da luu gamma_comparison.png, G0(gamma=0)=0, G0(1)=10")
    plt.close()

if __name__ == "__main__":
    main()
