"""
Bai 34. Danh gia policy bang simulation.
"""
import gymnasium as gym
import numpy as np

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    successes=0
    rewards=[]
    lengths=[]
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed+ep)
        done=False
        length=0
        total=0
        for _ in range(100):
            s=obs
            action = int(policy[s])
            obs, reward, terminated, truncated, info = env.step(action)
            total+=reward
            length+=1
            if terminated or truncated:
                if reward==1: successes+=1
                break
        rewards.append(total)
        lengths.append(length)
    return {
        "success_rate": successes/n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    # random policy
    rng=np.random.default_rng(0)
    rand_pol=np.array([rng.integers(0,4) for _ in range(16)])
    print("Random:", evaluate_policy_by_simulation(env, rand_pol, 1000, 0))

    # optimal via VI (tam dung)
    # import tu bai32
    from bai32 import value_iteration
    from bai33 import greedy_policy_from_value
    V,_,_=value_iteration(env,0.99)
    opt=greedy_policy_from_value(env,V,0.99)
    print("Optimal VI:", evaluate_policy_by_simulation(env, opt, 1000, 0))

    # policy iteration
    from bai29 import policy_iteration
    pi,_,_=policy_iteration(env,0.99)
    print("Optimal PI:", evaluate_policy_by_simulation(env, pi, 1000, 0))
    env.close()

if __name__ == "__main__":
    main()
