"""
Bai 24. Hien thi FrozenLake dang text
Tao render_mode="ansi", reset va in env.render(), quan sat Start, Frozen, Hole, Goal.
"""
import gymnasium as gym


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    obs, info = env.reset(seed=42)
    rendered = env.render()
    print(rendered)
    print(f"Initial observation (state): {obs}")
    print("\nChu thich:")
    print("  S = Start (vi tri bat dau)")
    print("  F = Frozen (bang dong, an toan)")
    print("  H = Hole (ho, roi xuong la thua)")
    print("  G = Goal (dich, thang)")
    print("\nVi tri agent duoc lam noi bat (thuong highlight do).")

    env.close()


if __name__ == "__main__":
    main()
