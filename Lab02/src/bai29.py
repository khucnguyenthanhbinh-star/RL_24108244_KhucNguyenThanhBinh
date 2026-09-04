"""
Bai 29. Policy Iteration.
"""
import gymnasium as gym
import numpy as np

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V=np.zeros(env.observation_space.n)
    for _ in range(max_iterations):
        delta=0
        new_V=np.zeros_like(V)
        for s in range(env.observation_space.n):
            v=sum(policy[s,a]*q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n))
            new_V[s]=v
            delta=max(delta, abs(v-V[s]))
        V=new_V
        if delta<theta: break
    return V

def greedy_policy_from_value(env,V,gamma):
    pol=np.zeros(env.observation_space.n,dtype=int)
    for s in range(env.observation_space.n):
        pol[s]=int(np.argmax([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)]))
    return pol

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    nS,nA=env.observation_space.n, env.action_space.n
    policy=np.ones((nS,nA))/nA
    for it in range(1, max_iterations+1):
        V=policy_evaluation(env, policy, gamma, theta)
        new_pi_det=greedy_policy_from_value(env,V,gamma)
        # Chuyen sang stochastic de so sanh
        new_policy=np.zeros((nS,nA))
        for s in range(nS):
            new_policy[s, new_pi_det[s]]=1.0
        if np.array_equal(np.argmax(policy,axis=1), new_pi_det):
            # policy stable
            return new_pi_det, V, it
        policy=new_policy
    return np.argmax(policy,axis=1), V, max_iterations

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    pi, V, it = policy_iteration(env, gamma=0.99)
    print(f"Policy Iteration converged after {it} iterations")
    print("Policy:", pi)
    print("V:", V)
    env.close()

if __name__ == "__main__":
    main()
