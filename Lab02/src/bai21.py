"""
Bai 21. q_from_v: Q(s,a)=sum p*[r+gamma*V(s')].
"""
import gymnasium as gym
import numpy as np

def q_from_v(env, V, state, action, gamma):
    Q = 0.0
    for prob, nxt, reward, terminated in env.unwrapped.P[state][action]:
        # Neu terminated thi V(next)=0? Nhung trong FrozenLake terminated chi khi vao H/G, V(G)=0, van cong thuc chung
        Q += prob * (reward + gamma * V[nxt] * (0 if terminated else 1) + gamma * V[nxt] * (1 if not terminated else 0))
        # Don gian: Q += prob * (reward + gamma * V[nxt])  # vi V[terminal]=0 sau hoi tu
        # De chinh xac theo cong thuc: khong nhan (1-terminated), vi V terminal van tinh nhung se la 0
    # Sua: dung cong thuc chuan
    Q = 0.0
    for prob, nxt, reward, terminated in env.unwrapped.P[state][action]:
        Q += prob * (reward + gamma * V[nxt])
    return Q

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    V = np.zeros(env.observation_space.n)
    print("Q(0, RIGHT) with V=0:", q_from_v(env, V, 0, 2, 0.99))
    # V gia tri ngau nhien
    V = np.random.rand(env.observation_space.n)
    print("Q(5,a) with random V:", [q_from_v(env, V, 5, a, 0.99) for a in range(4)])
    env.close()

if __name__ == "__main__":
    main()
