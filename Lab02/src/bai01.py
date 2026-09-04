"""
Bai 1. Tao transition matrix 3x3 cho Sunny, Cloudy, Rainy. Moi hang tong =1.
"""
import numpy as np

# 0: Sunny, 1: Cloudy, 2: Rainy
P = np.array([
    [0.7, 0.2, 0.1],  # Tu Sunny: 70% Sunny, 20% Cloudy, 10% Rainy
    [0.3, 0.4, 0.3],  # Tu Cloudy
    [0.2, 0.5, 0.3],  # Tu Rainy
])

def main():
    print("Transition matrix P (3x3):")
    print(P)
    print("\nHang 0 (Sunny):", P[0], " tong=", P[0].sum())
    print("Hang 1 (Cloudy):", P[1], " tong=", P[1].sum())
    print("Hang 2 (Rainy):", P[2], " tong=", P[2].sum())
    states = ["Sunny", "Cloudy", "Rainy"]
    for i, s in enumerate(states):
        for j, t in enumerate(states):
            print(f"  P({s}->{t}) = {P[i,j]:.2f}")

if __name__ == "__main__":
    main()
