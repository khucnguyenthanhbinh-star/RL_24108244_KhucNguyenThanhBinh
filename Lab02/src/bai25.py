"""
Bai 25. Theo doi hoi tu: deltas.
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def policy_evaluation_with_deltas(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V=np.zeros(env.observation_space.n)
    deltas=[]
    for i in range(max_iterations):
        new_V=np.zeros_like(V)
        delta=0
        for s in range(env.observation_space.n):
            v=sum(policy[s,a]*q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n))
            new_V[s]=v
            delta=max(delta, abs(v-V[s]))
        deltas.append(delta)
        V=new_V
        if delta < theta:
            break
    return V, deltas

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    policy=np.ones((16,4))/4
    V, deltas=policy_evaluation_with_deltas(env,policy)
    print(f"Iterations: {len(deltas)}, final delta {deltas[-1]:.2e}")
    plt.figure(figsize=(8,5))
    plt.plot(deltas); plt.yscale('log')
    plt.title("Policy Evaluation - Delta theo iteration")
    plt.xlabel("Iteration"); plt.ylabel("Delta (log)"); plt.grid(True, alpha=0.3)
    os.makedirs("Lab02/figures", exist_ok=True)
    plt.savefig("Lab02/figures/policy_iteration_convergence.png", dpi=150, bbox_inches='tight')
    for p in ["figures/policy_iteration_convergence.png","../figures/policy_iteration_convergence.png"]:
        try: os.makedirs(os.path.dirname(p), exist_ok=True); plt.savefig(p, dpi=150, bbox_inches='tight')
        except: pass
    print("Da luu policy_iteration_convergence.png (dung tam cho PE, VI se luu rieng)")
    plt.close()
    # Luu dung file yeu cau bai25 la policy_evaluation convergence
    plt.figure(figsize=(8,5))
    plt.plot(deltas); plt.yscale('log')
    plt.title("Policy Evaluation Convergence")
    plt.xlabel("Iteration"); plt.ylabel("Delta"); plt.grid(True, alpha=0.3)
    # Ghi de len file value_iteration_convergence tam? De bai25 chua ro, ta luu them
    try:
        plt.savefig("Lab02/figures/value_iteration_convergence.png", dpi=150, bbox_inches='tight')
    except: pass
    plt.close()
    env.close()

if __name__ == "__main__":
    main()
