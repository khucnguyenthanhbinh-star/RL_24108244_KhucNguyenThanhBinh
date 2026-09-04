# Lab02 - Markov Decision Process và Dynamic Programming

## Thông tin sinh viên

- Họ tên: Khúc Nguyễn Thanh Bình
- MSSV: 24108244
- Lớp: EEE-AI
- GitHub username: khucnguyenthanhbinh-star

## Mục tiêu

- Biểu diễn Markov chain bằng ma trận chuyển, kiểm tra hợp lệ, mô phỏng
- Tính Return với discount gamma, phân tích ảnh hưởng gamma
- Mô hình hóa MDP rời rạc, phân biệt deterministic/stochastic policy
- Đọc model FrozenLake-v1 qua `env.unwrapped.P`
- Cài đặt Bellman backup, Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration
- Đánh giá và so sánh thuật toán DP bằng simulation

## Cấu trúc thư mục

```
Lab02/
├── README.md
├── requirements.txt
├── src/bai01.py ... bai36.py, mdp_utils.py, main.py
├── notebooks/Lab02_24108244_KhucNguyenThanhBinh.ipynb
├── figures/markov_distribution.png, gamma_comparison.png, value_iteration_convergence.png, policy_iteration_convergence.png, algorithm_comparison.png
└── data/README.md
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r Lab02/requirements.txt
# hoac
pip install "gymnasium[toy-text]==1.3.0" numpy matplotlib jupyter
```

## Cách chạy

```bash
cd Lab02
pip install -r requirements.txt
python src/bai01.py
python src/bai24.py
python src/bai29.py
python src/bai32.py
python src/main.py
jupyter notebook notebooks/Lab02_24108244_KhucNguyenThanhBinh.ipynb
```

## Thuật toán đã cài đặt

### Policy Evaluation
Iterative Policy Evaluation với Bellman expectation backup `V(s)=sum_a pi(a|s) sum p(s'|s,a)[r+gamma V(s')]`, dừng khi delta < theta.

### Policy Iteration
Lặp Policy Evaluation → Greedy Improvement cho tới khi policy ổn định. Thường hội tụ sau 3-5 vòng với FrozenLake 4x4.

### Value Iteration
Cập nhật `V(s)=max_a sum p[r+gamma V(s')]` cho mọi state, trích optimal policy bằng argmax Q.

## Kết quả FrozenLake

- Môi trường: FrozenLake-v1 4x4 is_slippery=True, 16 states, 4 actions
- Value Iteration: hội tụ ~1400 iteration, success rate ~0.74
- Policy Iteration: hội tụ ~6 policy iteration (khoảng 100-200 evaluation sweep), success rate ~0.74
- Random policy: success ~0.01

## So sánh Value Iteration và Policy Iteration

| Thuật toán | Số vòng lặp | Thời gian | Success rate | Mean reward |
|---|---|---|---|---|
| Value Iteration | ~1400 | ~0.05s | 0.74 | 0.74 |
| Policy Iteration | 6 (với ~150 sweep) | ~0.03s | 0.74 | 0.74 |

Xem chi tiết tại `figures/algorithm_comparison.png` và convergence plots.

## Nhận xét

- DP cần biết model chuyển `P`, không áp dụng trực tiếp nếu không biết model.
- gamma gần 1 làm reward xa quan trọng hơn, gamma=0 chỉ quan tâm immediate reward.
- FrozenLake stochastic (is_slippery) làm policy phức tạp hơn deterministic.
- Value Iteration đơn giản nhưng nhiều sweep, Policy Iteration ít vòng lặp ngoài nhưng mỗi vòng tốn Evaluation.

## Tài liệu tham khảo

- https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- Sutton & Barto - Reinforcement Learning: An Introduction (Chương 3-4)
- Farama Gymnasium docs
