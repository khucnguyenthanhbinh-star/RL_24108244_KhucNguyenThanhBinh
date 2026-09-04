"""
Bai 26. Greedy policy tu V.
"""
import gymnasium as gym
import numpy as np

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def greedy_policy_from_value(env, V, gamma=0.99):
    nS, nA = env.observation_space.n, env.action_space.n
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_vals = [q_from_v(env,V,s,a,gamma) for a in range(nA)]
        policy[s] = int(np.argmax(q_vals))
    return policy

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V=np.zeros(16)
    pol=greedy_policy_from_value(env,V)
    print("Greedy from V=0:", pol)
    # Sau mot vai evaluation se khac
    env.close()

if __name__ == "__main__":
    main()
