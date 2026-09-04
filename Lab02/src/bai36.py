"""
Bai 36. Mini-project DP Solver hoan chinh cho FrozenLake.
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

ACTION_SYMBOLS={0:"←",1:"↓",2:"→",3:"↑"}

def create_environment(map_name="4x4", is_slippery=True):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

def get_transition_model(env):
    return env.unwrapped.P

def q_from_v(env, V, state, action, gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[state][action])

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V=np.zeros(env.observation_space.n)
    for i in range(max_iterations):
        delta=0
        new_V=np.zeros_like(V)
        for s in range(env.observation_space.n):
            v=sum(policy[s,a]*q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n))
            new_V[s]=v
            delta=max(delta, abs(v-V[s]))
        V=new_V
        if delta<theta:
            return V, i+1
    return V, max_iterations

def greedy_policy_from_value(env, V, gamma=0.99):
    pol=np.zeros(env.observation_space.n,dtype=int)
    for s in range(env.observation_space.n):
        pol[s]=int(np.argmax([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)]))
    return pol

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    nS,nA=env.observation_space.n, env.action_space.n
    policy=np.ones((nS,nA))/nA
    for it in range(1,max_iterations+1):
        V,_=policy_evaluation(env, policy, gamma, theta)
        new_pi=greedy_policy_from_value(env,V,gamma)
        old_pi=np.argmax(policy,axis=1)
        if np.array_equal(old_pi, new_pi):
            return new_pi, V, it
        policy=np.zeros((nS,nA))
        for s in range(nS): policy[s,new_pi[s]]=1
    return new_pi, V, max_iterations

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    V=np.zeros(env.observation_space.n)
    deltas=[]
    for i in range(max_iterations):
        new_V=np.zeros_like(V)
        delta=0
        for s in range(env.observation_space.n):
            new_V[s]=np.max([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)])
            delta=max(delta, abs(new_V[s]-V[s]))
        deltas.append(delta)
        V=new_V
        if delta<theta:
            return V, i+1, deltas
    return V, max_iterations, deltas

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    succ=0; rewards=[]; lengths=[]
    for ep in range(n_episodes):
        obs,_=env.reset(seed=seed+ep)
        total=0; length=0
        for _ in range(100):
            obs,reward,term,trunc,_=env.step(int(policy[obs]))
            total+=reward; length+=1
            if term or trunc:
                if reward==1: succ+=1
                break
        rewards.append(total); lengths.append(length)
    return {"success_rate":succ/n_episodes,"mean_reward":float(np.mean(rewards)),"mean_length":float(np.mean(lengths)),"min_length":int(np.min(lengths)),"max_length":int(np.max(lengths))}

def print_policy(env, policy):
    desc=env.unwrapped.desc
    for r in range(4):
        row=""
        for c in range(4):
            s=r*4+c
            cell=desc[r,c].decode()
            if cell=='H': row+=" H "
            elif cell=='G': row+=" G "
            else: row+=f" {ACTION_SYMBOLS[policy[s]]} "
        print(row)

def plot_convergence(deltas, title, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(deltas); plt.yscale('log')
    plt.title(title); plt.xlabel("Iteration"); plt.ylabel("Delta (log)"); plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

def main():
    print("=== Mini-project DP Solver FrozenLake 4x4 is_slippery=True ===")
    env=create_environment("4x4", True)
    gamma=0.99; theta=1e-8
    print(f"n_states {env.observation_space.n}, n_actions {env.action_space.n}, gamma {gamma}, theta {theta}")

    # Value Iteration
    start=perf_counter()
    V_vi, it_vi, deltas_vi = value_iteration(env, gamma, theta)
    time_vi=perf_counter()-start
    pi_vi=greedy_policy_from_value(env,V_vi,gamma)
    eval_vi=evaluate_policy_by_simulation(env, pi_vi, 1000, 42)
    print(f"\nValue Iteration: {it_vi} it, time {time_vi:.4f}s, success {eval_vi['success_rate']:.3f}, mean reward {eval_vi['mean_reward']:.3f}")
    print("V table:")
    print(V_vi.reshape(4,4))
    print("Policy:")
    print_policy(env, pi_vi)

    # Policy Iteration
    start=perf_counter()
    pi_pi, V_pi, it_pi = policy_iteration(env, gamma, theta)
    time_pi=perf_counter()-start
    eval_pi=evaluate_policy_by_simulation(env, pi_pi, 1000, 42)
    print(f"\nPolicy Iteration: {it_pi} policy it, time {time_pi:.4f}s, success {eval_pi['success_rate']:.3f}")
    print("Policy PI:")
    print_policy(env, pi_pi)

    # Random baseline
    rng=np.random.default_rng(0)
    rand_pi=np.array([rng.integers(0,4) for _ in range(16)])
    eval_rand=evaluate_policy_by_simulation(env, rand_pi, 1000, 0)
    print(f"\nRandom: success {eval_rand['success_rate']:.3f}")

    # Comparison
    print(f"\nComparison: VI {time_vi:.4f}s {it_vi}it vs PI {time_pi:.4f}s {it_pi}it")

    # Plots
    plot_convergence(deltas_vi, "Value Iteration Convergence", "Lab02/figures/value_iteration_convergence.png")
    # PI convergence: can log policy iteration deltas (tam)
    # Already have algorithm_comparison from bai35
    env.close()

if __name__ == "__main__":
    main()
