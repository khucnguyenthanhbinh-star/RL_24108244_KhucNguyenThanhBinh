"""
Bai 30. Kiem tra policy stability.
"""
import gymnasium as gym
import numpy as np

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V=np.zeros(env.observation_space.n)
    while True:
        delta=0
        new_V=np.zeros_like(V)
        for s in range(env.observation_space.n):
            v=sum(policy[s,a]*q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n))
            new_V[s]=v
            delta=max(delta, abs(v-V[s]))
        V=new_V
        if delta<theta: break
    return V

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    nS,nA=16,4
    policy=np.ones((nS,nA))/nA
    for it in range(1,100):
        V=policy_evaluation(env, policy)
        new_pi=np.zeros(nS,dtype=int)
        for s in range(nS):
            new_pi[s]=int(np.argmax([q_from_v(env,V,s,a,0.99) for a in range(nA)]))
        old_pi = np.argmax(policy,axis=1)
        policy_stable = np.array_equal(old_pi, new_pi)
        print(f"Iteration {it}: stable={policy_stable}, changed={np.sum(old_pi!=new_pi)}")
        if policy_stable:
            print(f"Policy Iteration converged after {it} iterations.")
            break
        # Update policy
        new_policy=np.zeros((nS,nA))
        for s in range(nS): new_policy[s,new_pi[s]]=1
        policy=new_policy
    env.close()

if __name__ == "__main__":
    main()
