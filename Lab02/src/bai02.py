"""
Bai 2. Kiem tra transition matrix: vuong, [0,1], tong hang ~1.
"""
import numpy as np

def validate_transition_matrix(P, tol=1e-10):
    # 1. Ma tran vuong
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    # 2. Moi phan tu [0,1]
    if np.any(P < -tol) or np.any(P > 1 + tol):
        return False
    # 3. Tong moi hang xap xi 1
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False
    return True

def main():
    P_valid = np.array([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.5,0.3]])
    P_invalid1 = np.array([[0.5,0.5],[0.5,0.5]])  # vuong nhung 2x2 ok test
    P_invalid2 = np.array([[0.7,0.2,0.1],[0.3,0.4,0.2],[0.2,0.5,0.3]])  # hang 1 tong 0.9
    P_invalid3 = np.array([[1.2,-0.1, -0.1],[0.3,0.4,0.3],[0.2,0.5,0.3]])

    print("P_valid:", validate_transition_matrix(P_valid))
    print("P 2x2 (van vuong):", validate_transition_matrix(P_invalid1))
    print("P tong hang sai:", validate_transition_matrix(P_invalid2))
    print("P phan tu ngoai [0,1]:", validate_transition_matrix(P_invalid3))

if __name__ == "__main__":
    main()
