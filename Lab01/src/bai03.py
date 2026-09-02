"""
Bai 3. Kham pha action space
In env.action_space va tu dong xac dinh so action.
"""
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    print(f"Action space: {env.action_space}")

    # Tu dong xac dinh so action, khong gan hang so
    if hasattr(env.action_space, "n"):
        n_actions = env.action_space.n
        print(f"Number of actions: {n_actions}")
        print(f"Action space type: Discrete({n_actions})")
        print("Valid actions: ", list(range(n_actions)))
    elif hasattr(env.action_space, "shape"):
        print(f"Action space shape: {env.action_space.shape}")
        print(f"Action space low: {env.action_space.low}")
        print(f"Action space high: {env.action_space.high}")
    else:
        print("Unknown action space type")

    env.close()


if __name__ == "__main__":
    main()
