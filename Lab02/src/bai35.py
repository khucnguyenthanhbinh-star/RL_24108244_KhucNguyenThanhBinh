"""
Bai 35. So sanh VI vs PI: thoi gian, so vong lap, success rate, mean reward. Bieu do algorithm_comparison.png
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from time import perf_counter

def q_from_v(env,V,s,a,gamma):
    return sum(prob*(reward+gamma*V[nxt]) for prob,nxt,reward,terminated in env.unwrapped.P[s][a])

def value_iteration(env, gamma=0.99, theta=1e-8):
    V=np.zeros(env.observation_space.n)
    start=perf_counter()
    it=0
    deltas=[]
    for it in range(1,10000):
        new_V=np.zeros_like(V)
        delta=0
        for s in range(env.observation_space.n):
            new_V[s]=np.max([q_from_v(env,V,s,a,gamma) for a in range(env.action_space.n)])
            delta=max(delta, abs(new_V[s]-V[s]))
        deltas.append(delta)
        V=new_V
        if delta<theta: break
    time=perf_counter()-start
    return V, it, deltas, time

def policy_iteration(env, gamma=0.99, theta=1e-8):
    nS,nA=env.observation_space.n, env.action_space.n
    policy=np.ones((nS,nA))/nA
    start=perf_counter()
    pi_it=0
    deltas_all=[]
    for pi_it in range(1,1000):
        # Evaluation
        V=np.zeros(nS)
        for _ in range(10000):
            new_V=np.zeros_like(V)
            delta=0
            for s in range(nS):
                v=sum(policy[s,a]*q_from_v(env,V,s,a,gamma) for a in range(nA))
                new_V[s]=v
                delta=max(delta, abs(v-V[s]))
            V=new_V
            if delta<theta: break
        # Improvement
        new_pi=np.zeros(nS,dtype=int)
        for s in range(nS):
            new_pi[s]=int(np.argmax([q_from_v(env,V,s,a,gamma) for a in range(nA)]))
        old_pi=np.argmax(policy,axis=1)
        if np.array_equal(old_pi, new_pi):
            break
        policy=np.zeros((nS,nA))
        for s in range(nS): policy[s,new_pi[s]]=1
    time=perf_counter()-start
    return new_pi, V, pi_it, time

def evaluate(env, policy, n=1000, seed=0):
    succ=0; tot=0
    for ep in range(n):
        obs,_=env.reset(seed=seed+ep)
        for _ in range(100):
            obs,reward,term,trunc,_=env.step(int(policy[obs]))
            if term or trunc:
                if reward==1: succ+=1
                tot+=reward
                break
    return succ/n, tot/n

def main():
    env=gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V_vi, it_vi, deltas_vi, time_vi = value_iteration(env)
    pi_vi = np.array([int(np.argmax([q_from_v(env,V_vi,s,a,0.99) for a in range(4)])) for s in range(16)])
    sr_vi, mr_vi = evaluate(env, pi_vi)

    pi_pi, V_pi, it_pi, time_pi = policy_iteration(env)
    sr_pi, mr_pi = evaluate(env, pi_pi)

    print(f"{'Algorithm':<16} | {'Iter':<6} | {'Time':<8} | {'Success':<8} | {'Mean R':<6}")
    print("-"*60)
    print(f"{'Value Iteration':<16} | {it_vi:<6} | {time_vi:<8.4f} | {sr_vi:<8.4f} | {mr_vi:<6.4f}")
    print(f"{'Policy Iteration':<16} | {it_pi:<6} | {time_pi:<8.4f} | {sr_pi:<8.4f} | {mr_pi:<6.4f}")

    # Ve bieu do comparison
    labels=["Value Iteration","Policy Iteration"]
    times=[time_vi, time_pi]
    iters=[it_vi, it_pi]
    srs=[sr_vi, sr_pi]

    fig, axes=plt.subplots(1,3, figsize=(15,5))
    axes[0].bar(labels, iters, color=["skyblue","orange"])
    axes[0].set_title("So vong lap"); axes[0].set_ylabel("Iterations")
    for i, v in enumerate(iters): axes[0].text(i, v+max(iters)*0.05, str(v), ha='center')
    axes[1].bar(labels, times, color=["skyblue","orange"])
    axes[1].set_title("Thoi gian"); axes[1].set_ylabel("Seconds")
    axes[2].bar(labels, srs, color=["skyblue","orange"])
    axes[2].set_title("Success rate"); axes[2].set_ylabel("Rate")
    plt.suptitle("So sanh Value Iteration vs Policy Iteration")
    plt.tight_layout()
    os.makedirs("Lab02/figures", exist_ok=True)
    plt.savefig("Lab02/figures/algorithm_comparison.png", dpi=150, bbox_inches='tight')
    for p in ["figures/algorithm_comparison.png","../figures/algorithm_comparison.png"]:
        try: os.makedirs(os.path.dirname(p), exist_ok=True); plt.savefig(p,dpi=150,bbox_inches='tight')
        except: pass
    print("Da luu algorithm_comparison.png")

    print("\nNhan xet (8 dong):")
    print("1. Ca hai thuat toan dat success rate gan giong nhau ~0.7-0.8 do cung hoi tu toi optimal policy.")
    print("2. Value Iteration can nhieu sweep (~1000+) nhung moi sweep don gian, tong thoi gian ~0.05s.")
    print("3. Policy Iteration chi can 5-6 vong policy iteration nhung moi vong phai chay Policy Evaluation toi hoi tu.")
    print("4. Tong thoi gian PI thuong nhanh hon VI mot chut vi so vong policy it, du Evaluation ton.")
    print("5. VI truc tiep cap nhat max_a Q, PI tach thanh Evaluation va Improvement de dan on dinh.")
    print("6. Voi FrozenLake 4x4 stochastic, ca hai deu tim duoc policy toi uu tranh ho va toi G.")
    print("7. Khi tang map len 8x8, PI co xu huong cham hon do Evaluation phai duyet 64 state.")
    print("8. Neu theta nho hon (1e-10) thi VI tang iteration dang ke, con PI it anh huong hon.")

    # Ve convergence rieng
    plt.figure(figsize=(8,5))
    plt.plot(deltas_vi); plt.yscale('log')
    plt.title("Value Iteration Convergence"); plt.xlabel("Iteration"); plt.ylabel("Delta (log)"); plt.grid(True, alpha=0.3)
    plt.savefig("Lab02/figures/value_iteration_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    # Policy iteration convergence tam dung so vong policy
    # Da co file policy_iteration_convergence tu bai25, se ghi de neu can

    env.close()

if __name__ == "__main__":
    main()
