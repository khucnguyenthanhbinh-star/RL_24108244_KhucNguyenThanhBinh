"""
Bai 24. Iterative Policy Evaluation.
"""
import gymnasium as gym
import numpy as np

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V=np.zeros(env.observation_space.n)
    for i in range(max_iterations):
        delta=0
        new_V=np.zeros_like(V)
        for s in range(env.observation_space.n):
            v = sum(policy[s,a]*q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n))
            new_V[s]=v
            delta=max(delta, abs(v-V[s]))
        V=new_V
        if delta < theta:
            return V, i+1
    return V, max_iterations

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    policy=np.ones((16,4))/4
    V, it = policy_evaluation(env,policy,gamma=0.99)
    print(f"Converged in {it} iterations")
    print("V:", V)
    env.close()

if __name__ == "__main__":
    main()
