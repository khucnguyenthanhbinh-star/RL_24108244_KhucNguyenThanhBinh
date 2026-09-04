"""
Bai 32. Value Iteration hoan chinh.
"""
import gymnasium as gym
import numpy as np

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    V=np.zeros(env.observation_space.n)
    deltas=[]
    for i in range(max_iterations):
        new_V=np.zeros_like(V)
        delta=0
        for s in range(env.observation_space.n):
            q_vals=[q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)]
            new_V[s]=np.max(q_vals)
            delta=max(delta, abs(new_V[s]-V[s]))
        deltas.append(delta)
        V=new_V
        if delta < theta:
            return V, i+1, deltas
    return V, max_iterations, deltas

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V, it, deltas = value_iteration(env, gamma=0.99)
    print(f"Converged in {it} iterations, final delta {deltas[-1]:.2e}")
    print("V:", V)
    env.close()

if __name__ == "__main__":
    main()
