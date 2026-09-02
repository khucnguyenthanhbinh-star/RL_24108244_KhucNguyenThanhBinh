"""
Bai 1. Kiem tra moi truong Python
In Python version, Gymnasium version, NumPy version tu dong, khong nhap thu cong.
"""
import sys
import gymnasium
import numpy


def main():
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python full: {sys.version}")
    print(f"Gymnasium version: {gymnasium.__version__}")
    print(f"NumPy version: {numpy.__version__}")


if __name__ == "__main__":
    main()
