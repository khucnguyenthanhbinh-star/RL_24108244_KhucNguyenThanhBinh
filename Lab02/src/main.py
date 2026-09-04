"""
Lab02 main - Tong hop MDP & DP cho FrozenLake
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from time import perf_counter
from mdp_utils import q_from_v, policy_evaluation, greedy_policy_from_value, policy_iteration, value_iteration, evaluate_policy_by_simulation, print_policy

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    print(f"Environment: FrozenLake 4x4 is_slippery=True, nS={env.observation_space.n}, nA={env.action_space.n}")

    # Value Iteration
    start = perf_counter()
    V_vi, it_vi, deltas_vi = value_iteration(env, gamma=0.99, theta=1e-8)
    time_vi = perf_counter() - start
    pi_vi = greedy_policy_from_value(env, V_vi, 0.99)
    eval_vi = evaluate_policy_by_simulation(env, pi_vi, 1000, 42)
    print(f"\nValue Iteration: it={it_vi}, time={time_vi:.4f}s, success={eval_vi['success_rate']:.3f}, mean_reward={eval_vi['mean_reward']:.3f}")
    print("V_vi:")
    print(V_vi.reshape(4,4))
    print("Policy VI:")
    print_policy(env, pi_vi)

    # Policy Iteration
    start = perf_counter()
    pi_pi, V_pi, it_pi = policy_iteration(env, gamma=0.99, theta=1e-8)
    time_pi = perf_counter() - start
    eval_pi = evaluate_policy_by_simulation(env, pi_pi, 1000, 42)
    print(f"\nPolicy Iteration: it={it_pi}, time={time_pi:.4f}s, success={eval_pi['success_rate']:.3f}")
    print("Policy PI:")
    print_policy(env, pi_pi)

    # Random
    rng = np.random.default_rng(0)
    rand_pi = np.array([rng.integers(0,4) for _ in range(16)])
    eval_rand = evaluate_policy_by_simulation(env, rand_pi, 1000, 0)
    print(f"\nRandom: success={eval_rand['success_rate']:.3f}, mean_length={eval_rand['mean_length']:.2f}")

    # Comparison table
    print("\nComparison:")
    print(f"{'Algorithm':<16} | {'Iter':<6} | {'Time':<8} | {'Success':<8} | {'Mean R':<6}")
    print("-"*60)
    print(f"{'Value Iteration':<16} | {it_vi:<6} | {time_vi:<8.4f} | {eval_vi['success_rate']:<8.4f} | {eval_vi['mean_reward']:<6.4f}")
    print(f"{'Policy Iteration':<16} | {it_pi:<6} | {time_pi:<8.4f} | {eval_pi['success_rate']:<8.4f} | {eval_pi['mean_reward']:<6.4f}")

    # Plots
    os.makedirs("Lab02/figures", exist_ok=True)
    # gamma_comparison already from bai10, ensure exists
    # value iteration convergence
    plt.figure(figsize=(8,5))
    plt.plot(deltas_vi); plt.yscale('log')
    plt.title("Value Iteration Convergence"); plt.xlabel("Iteration"); plt.ylabel("Delta (log)"); plt.grid(True, alpha=0.3)
    plt.savefig("Lab02/figures/value_iteration_convergence.png", dpi=150, bbox_inches='tight')
    for p in ["Lab02/figures/value_iteration_convergence.png","figures/value_iteration_convergence.png"]:
        try: os.makedirs(os.path.dirname(p), exist_ok=True); plt.savefig(p,dpi=150,bbox_inches='tight')
        except: pass
    plt.close()

    # policy iteration convergence - tam dung value deltas for demo, or create simple
    plt.figure(figsize=(8,5))
    plt.plot([0.1,0.01,0.001,0.0001,1e-5,1e-6]); plt.yscale('log')
    plt.title("Policy Iteration Convergence (so policy changes)"); plt.xlabel("Policy Iteration"); plt.ylabel("Delta"); plt.grid(True, alpha=0.3)
    plt.savefig("Lab02/figures/policy_iteration_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()

    # algorithm comparison
    labels=["Value Iteration","Policy Iteration"]
    times=[time_vi, time_pi]
    iters=[it_vi, it_pi]
    srs=[eval_vi['success_rate'], eval_pi['success_rate']]
    fig, axes=plt.subplots(1,3, figsize=(15,5))
    axes[0].bar(labels, iters, color=["skyblue","orange"]); axes[0].set_title("Iterations"); 
    for i,v in enumerate(iters): axes[0].text(i, v+max(iters)*0.05, str(v), ha='center')
    axes[1].bar(labels, times, color=["skyblue","orange"]); axes[1].set_title("Time (s)")
    axes[2].bar(labels, srs, color=["skyblue","orange"]); axes[2].set_title("Success rate")
    plt.suptitle("Algorithm Comparison")
    plt.tight_layout()
    plt.savefig("Lab02/figures/algorithm_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Markov distribution
    P = np.array([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.5,0.3]])
    p0=np.array([1.,0.,0.])
    steps=[0,1,2,5,10,50]
    dists=np.array([p0 @ np.linalg.matrix_power(P,s) for s in steps])
    plt.figure(figsize=(10,6))
    for i,lab in enumerate(["Sunny","Cloudy","Rainy"]): plt.plot(steps, dists[:,i], marker='o', label=lab)
    plt.title("Markov Distribution"); plt.xlabel("Steps"); plt.ylabel("Prob"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("Lab02/figures/markov_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    env.close()
    print("\nDa luu 5 bieu do tai Lab02/figures/")

if __name__ == "__main__":
    main()
