# Bai thuc hanh so 1 - Lam quen voi Gymnasium

Ho ten: Khuc Nguyen Thanh Binh
MSSV: 24108244
Lop: EEE-AI
GitHub username: khucnguyenthanhbinh-star
Repository URL: https://github.com/khucnguyenthanhbinh-star/RL_24108244_KhucNguyenThanhBinh

Python version: 3.13.14
Gymnasium version: 1.3.0
NumPy version: 2.5.2
Matplotlib version: 3.11.1

## Cach cai dat

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
# hoac
pip install "gymnasium[classic-control,toy-text]==1.3.0" numpy matplotlib jupyter
```

Kiem tra:
```bash
python --version
pip show gymnasium
```

## Cach chay tung bai

```bash
cd Lab01
python src/bai01.py
python src/bai02.py
...
python src/bai36.py
python src/main.py
python src/migration_gym_to_gymnasium.py
python src/starter.py
```

## Cach chay chuong trinh tong hop

```bash
python src/main.py
jupyter notebook notebooks/Lab01_24108244_KhucNguyenThanhBinh.ipynb
```

## Mo ta ket qua

- Bai 1-6: Kiem tra moi truong, action_space Discrete(2), observation_space Box(4,)
- Bai 7-12: Random agent CartPole, episode trung binh ~20-50 reward
- Bai 13-18: 100 episode thong ke mean/std, ve reward_cartpole.png va moving_average.png (window=10)
- Bai 19-22: Seed dam bao tai lap, experiment() voi 5 seed
- Bai 23-28: FrozenLake 16 states/4 actions, success_rate ~0.05-0.15 random (stochastic) vs ~0.7 deterministic
- Bai 29-32: Heuristic angle_based vuot random 2-3 lan
- Bai 33-35: So sanh 3 agent 500 episode, ve comparison_agents.png
- Bai 36: Mini-project CartPole hoan chinh 500 episode

## Kho khan gap phai

- API Gymnasium moi tra ve 5 gia tri o step() khac voi gym cu
- terminated vs truncated can phan biet ro
- Seed cho ca env.reset() va action_space

## Ket luan

Da hieu vong lap Agent-Environment co ban: observation,info=env.reset(); action=policy(obs); next_obs,reward,terminated,truncated,info=env.step(action); lap den khi terminated or truncated. Biet to chuc thi nghiem RL, thong ke reward va su dung GitHub.

## Tai lieu tham khao

- https://gymnasium.farama.org/
- https://github.com/Farama-Foundation/Gymnasium
