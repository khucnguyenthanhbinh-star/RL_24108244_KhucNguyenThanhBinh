"""
Bai 27. Hien thi policy 4x4.
"""
import gymnasium as gym
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ACTION_SYMBOLS = {0:"←",1:"↓",2:"→",3:"↑"}

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def greedy_policy_from_value(env,V,gamma=0.99):
    pol=np.zeros(env.observation_space.n,dtype=int)
    for s in range(env.observation_space.n):
        pol[s]=int(np.argmax([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)]))
    return pol

def print_frozenlake_policy(env, policy):
    # policy la array size 16
    desc = env.unwrapped.desc  # 4x4 array of bytes
    for r in range(4):
        row=""
        for c in range(4):
            s=r*4+c
            cell=desc[r,c].decode()
            if cell=='H':
                row+=" H "
            elif cell=='G':
                row+=" G "
            else:
                row+=f" {ACTION_SYMBOLS[policy[s]]} "
        print(row)

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V=np.zeros(16)
    # Dung value sau policy evaluation gan dung de co policy hon
    pol=greedy_policy_from_value(env,V)
    print("Policy tu V=0:")
    print_frozenlake_policy(env,pol)
    # Vi du policy tot hon sau VI
    env.close()

if __name__ == "__main__":
    main()
