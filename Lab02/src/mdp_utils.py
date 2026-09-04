"""
mdp_utils.py - Chua cac ham DP dung chung cho Lab02, import thay vi copy-paste.
"""
import gymnasium as gym
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ACTION_SYMBOLS = {0:"←",1:"↓",2:"→",3:"↑"}
ACTION_NAMES = {0:"LEFT",1:"DOWN",2:"RIGHT",3:"UP"}

def q_from_v(env, V, state, action, gamma):
    """Q(s,a)=sum p*(r+gamma*V(s'))"""
    return sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward, terminated in env.unwrapped.P[state][action])

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    for _ in range(max_iterations):
        delta = 0
        new_V = np.zeros_like(V)
        for s in range(env.observation_space.n):
            v = sum(policy[s, a] * q_from_v(env, V, s, a, gamma) for a in range(env.action_space.n))
            new_V[s] = v
            delta = max(delta, abs(v - V[s]))
        V = new_V
        if delta < theta:
            break
    return V

def greedy_policy_from_value(env, V, gamma=0.99):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        q_vals = [q_from_v(env, V, s, a, gamma) for a in range(env.action_space.n)]
        policy[s] = int(np.argmax(q_vals))
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    nS, nA = env.observation_space.n, env.action_space.n
    policy = np.ones((nS, nA)) / nA
    for it in range(1, max_iterations+1):
        V = policy_evaluation(env, policy, gamma, theta)
        new_pi = greedy_policy_from_value(env, V, gamma)
        old_pi = np.argmax(policy, axis=1)
        if np.array_equal(old_pi, new_pi):
            return new_pi, V, it
        policy = np.zeros((nS, nA))
        for s in range(nS):
            policy[s, new_pi[s]] = 1.0
    return new_pi, V, max_iterations

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    deltas = []
    for i in range(max_iterations):
        new_V = np.zeros_like(V)
        delta = 0
        for s in range(env.observation_space.n):
            new_V[s] = np.max([q_from_v(env, V, s, a, gamma) for a in range(env.action_space.n)])
            delta = max(delta, abs(new_V[s] - V[s]))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            return V, i+1, deltas
    return V, max_iterations, deltas

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    succ = 0
    rewards = []
    lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        total = 0
        length = 0
        for _ in range(100):
            obs, reward, term, trunc, _ = env.step(int(policy[obs]))
            total += reward
            length += 1
            if term or trunc:
                if reward == 1:
                    succ += 1
                break
        rewards.append(total)
        lengths.append(length)
    return {
        "success_rate": succ / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }

def print_policy(env, policy):
    desc = env.unwrapped.desc
    for r in range(4):
        row = ""
        for c in range(4):
            s = r*4+c
            cell = desc[r,c].decode()
            if cell == 'H':
                row += " H "
            elif cell == 'G':
                row += " G "
            else:
                row += f" {ACTION_SYMBOLS[policy[s]]} "
        print(row)
