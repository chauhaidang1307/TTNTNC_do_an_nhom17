# Connect Four AI đa cấp độ

## 1. Đề tài

**Xây dựng và đánh giá tác tử AI đa cấp độ cho trò chơi Connect Four: từ tìm kiếm đối kháng đến học tăng cường.**

Trò chơi: Connect Four bàn chuẩn 7 cột × 6 hàng, hai người đi luân phiên, thả quân vào một cột, thắng khi có 4 quân liên tiếp và hòa khi bàn đầy.

Năm tác tử:

| Cấp | Tác tử | Mô tả |
|---|---|---|
| 1 | Random | Chọn ngẫu nhiên đều trong các nước hợp lệ; dùng làm mốc so sánh. |
| 2 | Heuristic | Đánh theo luật ưu tiên: thắng ngay, chặn nước thắng của đối thủ, tránh nước tạo cơ hội cho đối thủ, ưu tiên cột giữa. |
| 3 | Alpha-Beta | Tìm kiếm đối kháng có cắt tỉa, giới hạn theo độ sâu hoặc thời gian; có thể tắt cắt tỉa để so sánh. |
| 4 | MCTS (UCT) | Tìm kiếm cây Monte Carlo, giới hạn theo số mô phỏng hoặc thời gian. |
| 6 | Reinforcement Learning | Mạng Double DQN học bằng tự chơi (self-play). |

## 2. Môi trường
- Hệ điều hành: Windows 11.
- Python: 3.10 (đã kiểm tra với 3.10.11).
- Phần cứng đã dùng: NVIDIA GeForce RTX 3050 Laptop (4 GB), driver 617.42.

Cài đặt (chạy trong thư mục gốc, PowerShell):

1. Tạo và kích hoạt môi trường ảo bằng Python 3.10:
   ```
   py -3.10 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Cài PyTorch bản CUDA **trước**, riêng một lệnh (cài từ requirements sẽ ra bản CPU):
   ```
   python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
   ```
3. Cài các thư viện còn lại:
   ```
   python -m pip install -r requirements.txt
   ```
4. Kiểm tra GPU:
   ```
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```
   Kết quả đúng: `2.14.1+cu132 13.2 True`. Nếu thấy `+cpu`, `None` hoặc `False` thì đang dùng bản CPU, hãy gỡ torch và làm lại bước 2.

Máy không có GPU NVIDIA: dùng bản CPU của torch (cài `torch` mặc định từ PyPI). Các tác tử tìm kiếm và test chạy bình thường; việc train RL sẽ chậm hơn nhiều.

## 3. Cấu trúc thư mục
```
Connect4/
├── engine/          bàn cờ và luật
├── agents/          các tác tử
├── rl/              huấn luyện RL
├── arena/           giải đấu, ghi log
├── ui/              giao diện
├── tests/           pytest
├── configs/         cấu hình thí nghiệm
├── logs/            log thô
├── analysis/        notebook vẽ bảng, biểu đồ
├── setup_environment.ipynb
├── requirements.txt
└── README.md
```
