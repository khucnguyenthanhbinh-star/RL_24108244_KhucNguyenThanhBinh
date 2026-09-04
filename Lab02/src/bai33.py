"""
Bai 33. Trich xuat optimal policy sau Value Iteration.
"""
import gymnasium as gym
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ACTION_SYMBOLS={0:"←",1:"↓",2:"→",3:"↑"}

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def value_iteration(env, gamma=0.99, theta=1e-8):
    V=np.zeros(env.observation_space.n)
    while True:
        new_V=np.zeros_like(V)
        delta=0
        for s in range(env.observation_space.n):
            new_V[s]=np.max([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)])
            delta=max(delta, abs(new_V[s]-V[s]))
        V=new_V
        if delta<theta: break
    return V

def greedy_policy_from_value(env,V,gamma=0.99):
    pol=np.zeros(env.observation_space.n,dtype=int)
    for s in range(env.observation_space.n):
        pol[s]=int(np.argmax([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)]))
    return pol

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

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V=value_iteration(env)
    print("Optimal state values:")
    print(V.reshape(4,4))
    optimal_policy=greedy_policy_from_value(env,V,0.99)
    print("Optimal policy:", optimal_policy)
    print_policy(env, optimal_policy)
    env.close()

if __name__ == "__main__":
    main()
