"""
Bai 3. Tinh xac suat trang thai ke tiep: p0 * P.
"""
import numpy as np

P = np.array([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.5,0.3]])

def main():
    p0 = np.array([1.0, 0.0, 0.0])  # bat dau Sunny
    print("p0:", p0)
    print("P:", P, sep="\n")
    # p1 = p0 @ P
    p1 = p0 @ P
    print("p1 = p0 @ P:", p1)
    # Khong hard-code, tinh dong
    print(f"Sau 1 buoc: Sunny {p1[0]:.2f}, Cloudy {p1[1]:.2f}, Rainy {p1[2]:.2f}")

if __name__ == "__main__":
    main()
