"""
Bai 22. action_values: Q(s,a) cho moi a.
"""
import gymnasium as gym
import numpy as np

def q_from_v(env, V, state, action, gamma):
    Q=0.0
    for prob,nxt,reward,terminated in env.unwrapped.P[state][action]:
        Q+=prob*(reward+gamma*V[nxt])
    return Q

def action_values(env, V, state, gamma):
    return np.array([q_from_v(env,V,state,a,gamma) for a in range(env.action_space.n)])

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V=np.zeros(env.observation_space.n)
    print("Q(0):", action_values(env,V,0,0.99))
    print("Q(1):", action_values(env,V,1,0.99))
    env.close()

if __name__ == "__main__":
    main()
