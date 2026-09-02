"""
Bai 12. Khong su dung bien done tu API cu
Viet lai Bai 11, tu tao episode_finished = terminated or truncated, in rieng Termination/Truncation.
"""
import gymnasium as gym


def random_agent_no_done(env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = env.action_space.sample()
        # API moi tra ve 5 gia tri, KHONG co done
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1

        episode_finished = terminated or truncated
        if episode_finished:
            break

    # In rieng nguyen nhan
    if terminated:
        reason = "Termination (agent di den trang thai ket thuc theo ban chat bai toan - pole do hoac cart ra bien)"
    elif truncated:
        reason = "Truncation (episode bi dung boi gioi han ben ngoai - vuot max_steps=500)"
    else:
        reason = "Chua ket thuc (het max_steps nhung chua terminated/truncated)"

    return total_reward, episode_length, terminated, truncated, reason


def main():
    env = gym.make("CartPole-v1")
    total_reward, length, terminated, truncated, reason = random_agent_no_done(env)
    print(f"Total reward: {total_reward}")
    print(f"Episode length: {length}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Ly do ket thuc: {reason}")

    # Demo them 3 episode
    print("\nDemo 3 episode:")
    for i in range(3):
        r, l, term, trunc, rs = random_agent_no_done(env)
        print(f"  Episode {i+1}: reward={r}, length={l}, terminated={term}, truncated={trunc} -> {rs.split('(')[0].strip()}")

    env.close()


if __name__ == "__main__":
    main()
