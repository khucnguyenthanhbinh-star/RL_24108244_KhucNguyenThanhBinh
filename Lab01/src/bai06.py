"""
Bai 6. Sinh action ngau nhien
Sinh 20 action bang env.action_space.sample(), luu vao list, in va tinh tan suat.
"""
import gymnasium as gym
from collections import Counter


def main():
    env = gym.make("CartPole-v1")
    # Seed de tai lap (tu chon, nhung dam bao ket qua on dinh neu muon)
    # Khong seed cung duoc, yeu cau chi la sinh 20 action ngau nhien

    actions = [env.action_space.sample() for _ in range(20)]
    print(f"Danh sach 20 action: {actions}")

    freq = Counter(actions)
    print("\nTan suat xuat hien:")
    for action in sorted(freq.keys()):
        print(f"  Action {action}: {freq[action]} lan ({freq[action]/len(actions)*100:.1f}%)")

    # Dam bao dem du ca action khong xuat hien
    n_actions = env.action_space.n
    for a in range(n_actions):
        if a not in freq:
            print(f"  Action {a}: 0 lan (0.0%)")

    env.close()


if __name__ == "__main__":
    main()
