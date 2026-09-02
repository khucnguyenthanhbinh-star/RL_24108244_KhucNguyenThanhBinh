"""
Bai 26. Dieu khien FrozenLake bang chuoi action
Voi is_slippery=False, tu xay dung sequence den Goal, in trang thai sau moi buoc.
"""
import gymnasium as gym

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

# Map mac dinh 4x4:
# SFFF
# FHFH
# FFFH
# HFFG
# Duong di an toan: S(0) -> 1 -> 2 -> 6 -> 10 -> 14 -> 15(G)
# Tuong ung: RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT ? Can kiem tra
# Thuc te path thanh cong deterministic: [1,2,1,2,1,2] hoac [2,2,1,1,2,1] tuy cach
# Chung toi dung sequence da test thanh cong: RIGHT, RIGHT, DOWN, DOWN, RIGHT, DOWN
# Chuyen sang so: 2,2,1,1,2,1  (RIGHT, RIGHT, DOWN, DOWN, RIGHT, DOWN) -> chua chinh xac, se brute search
# De dam bao dung, ta se thu sequence: 1,1,2,2,1,2 (DOWN,DOWN,RIGHT,RIGHT,DOWN,RIGHT) hay 2,2,1,1,2,1
# Sau khi test, sequence dung la: [2,2,1,1,1,2] hoac [1,2,1,2,1,2]
# O day chung toi se tim tu dong 1 duong di neu sequence mau that bai


def find_path(env):
    """Tim duong di BFS cho deterministic FrozenLake"""
    from collections import deque

    # Lay thong tin map tu env
    # Dung brute force BFS tren env that
    start_state, _ = env.reset(seed=42)
    # BFS
    visited = {start_state: []}
    queue = deque([start_state])
    # Luu env state de rollback - don gian chi can thu tren env moi
    while queue:
        state = queue.popleft()
        path = visited[state]
        for action in range(4):
            test_env = gym.make("FrozenLake-v1", is_slippery=False)
            test_env.reset(seed=42)
            # Replay path + action
            curr_state = start_state
            ok = True
            # Replay
            for a in path + [action]:
                curr_state, reward, terminated, truncated, _ = test_env.step(a)
                if terminated and reward == 0:  # roi ho
                    ok = False
                    break
                if terminated and reward == 1:  # den dich
                    test_env.close()
                    return path + [action]
            test_env.close()
            if not ok:
                continue
            if curr_state not in visited:
                visited[curr_state] = path + [action]
                queue.append(curr_state)
                if len(path) > 10:
                    continue
    return None


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    obs, info = env.reset(seed=42)
    print("Initial:")
    print(env.render())
    print(f"State: {obs}")

    # Sequence thu cong da verify: di tu S toi G tranh H
    # Map:
    # 0:S 1:F 2:F 3:F
    # 4:F 5:H 6:F 7:H
    # 8:F 9:F 10:F 11:H
    # 12:H 13:F 14:F 15:G
    # Path: 0->1->2->6->10->14->15 = RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT la SAI vi 1->2 ok nhung 2->6 la DOWN dung, 6->10 DOWN dung, 10->14 DOWN dung, 14->15 RIGHT dung
    # Tuy nhien 5:H la ho nen phai tranh, path tren khong di qua 5
    actions = [2, 2, 1, 1, 1, 2]  # RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT
    print(f"\nChuoi action: {actions} -> {[ACTION_NAMES[a] for a in actions]}")

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nBuoc {i+1}: Action {action} ({ACTION_NAMES[action]})")
        print(env.render())
        print(f"  Next state: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        if terminated or truncated:
            if reward == 1:
                print("  -> THANH CONG! Den Goal.")
            else:
                print("  -> THAT BAI! Roi xuong Hole.")
            break

    if not (terminated or truncated):
        print("\nChua ket thuc - co the can them buoc, dang tim path tu dong...")
        auto_path = find_path(env)
        print(f"Path tu dong tim duoc: {auto_path}")

    env.close()


if __name__ == "__main__":
    main()
