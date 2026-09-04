"""
Bai 28. Mot buoc Policy Improvement: Evaluation -> Greedy -> so sanh.
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

def greedy_policy_from_value(env,V,gamma=0.99):
    pol=np.zeros(env.observation_space.n,dtype=int)
    for s in range(env.observation_space.n):
        pol[s]=int(np.argmax([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)]))
    return pol

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    nS, nA=16,4
    old_policy=np.ones((nS,nA))/nA  # uniform
    V=policy_evaluation(env, old_policy)
    # Chuyen old_policy sang deterministic de so sanh
    old_det=np.array([np.argmax(old_policy[s]) for s in range(nS)])
    new_policy=greedy_policy_from_value(env,V)
    changed=np.sum(old_det != new_policy)
    print(f"Old deterministic:", old_det)
    print(f"New greedy:", new_policy)
    print(f"So state doi action: {changed}/{nS}")
    env.close()

if __name__ == "__main__":
    main()
