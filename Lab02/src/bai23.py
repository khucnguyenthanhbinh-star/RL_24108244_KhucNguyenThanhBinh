"""
Bai 23. Mot sweep Policy Evaluation voi uniform random policy.
"""
import gymnasium as gym
import numpy as np

def q_from_v(env,V,state,action,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[state][action])

def policy_evaluation_sweep(env, policy, V, gamma):
    new_V = np.zeros_like(V)
    for s in range(env.observation_space.n):
        val=0.0
        for a in range(env.action_space.n):
            val+=policy[s,a]*q_from_v(env,V,s,a,gamma)
        new_V[s]=val
    return new_V

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    nS, nA = env.observation_space.n, env.action_space.n
    policy = np.ones((nS,nA))/nA
    V=np.zeros(nS)
    new_V=policy_evaluation_sweep(env,policy,V,0.99)
    print("V after 1 sweep:", new_V)
    env.close()

if __name__ == "__main__":
    main()
