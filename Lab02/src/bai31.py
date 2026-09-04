"""
Bai 31. Mot sweep Value Iteration: new_V[s]=max_a Q(s,a)
"""
import gymnasium as gym
import numpy as np

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def value_iteration_sweep(env, V, gamma):
    new_V=np.zeros_like(V)
    for s in range(env.observation_space.n):
        q_vals=[q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)]
        new_V[s]=np.max(q_vals)
    return new_V

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V=np.zeros(16)
    new_V=value_iteration_sweep(env,V,0.99)
    print("V after 1 sweep:", new_V)
    env.close()

if __name__ == "__main__":
    main()
