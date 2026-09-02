"""
Bai 21. Seed cho action_space
Tim cach seed env.action_space, sinh 20 action, chay 2 lan kiem tra chuoi co giong nhau khong.
"""
import gymnasium as gym


def sample_actions_with_seed(seed):
    env = gym.make("CartPole-v1")
    # Seed cho action_space
    env.action_space.seed(seed)
    actions = [env.action_space.sample() for _ in range(20)]
    env.close()
    return actions


def main():
    seed = 42
    actions1 = sample_actions_with_seed(seed)
    actions2 = sample_actions_with_seed(seed)

    print(f"Lan 1: {actions1}")
    print(f"Lan 2: {actions2}")
    print(f"Giong nhau khong? {actions1 == actions2}")

    # Thu khong seed
    env = gym.make("CartPole-v1")
    no_seed_actions1 = [env.action_space.sample() for _ in range(20)]
    no_seed_actions2 = [env.action_space.sample() for _ in range(20)]
    print(f"\nKhong seed lan 1: {no_seed_actions1}")
    print(f"Khong seed lan 2: {no_seed_actions2}")
    print(f"Giong nhau khong? {no_seed_actions1 == no_seed_actions2} (thuong la khac)")
    env.close()

    # Ket luan: seed action_space giup chuoi action tai lap duoc


if __name__ == "__main__":
    main()
