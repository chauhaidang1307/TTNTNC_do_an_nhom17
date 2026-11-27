# Connect Four AI đa cấp độ

**Xây dựng và đánh giá tác tử AI đa cấp độ cho trò chơi Connect Four: từ tìm kiếm đối kháng đến học tăng cường.**

Đồ án cuối kỳ, học phần *Trí tuệ nhân tạo nâng cao*. Mã nguồn Python tự cài đặt: game engine, năm tác tử (Random, Heuristic, Alpha-Beta, MCTS-UCT, Double DQN), hệ thống giải đấu có ghi log, giao diện người–máy và máy–máy, test tự động, notebook phân tích.

## Thông tin đồ án

| | |
|---|---|
| Học phần | Trí tuệ nhân tạo nâng cao |
| Giảng viên | Đỗ Như Tài |
| Nhóm | 17 |
| Tác giả | Châu Hải Đăng, MSSV 3123410075 |
| GitHub | https://github.com/chauhaidang1307/TTNTNC_do_an_nhom17 |
| Nhật ký thí nghiệm (Google Sheets) | https://docs.google.com/spreadsheets/d/1j2_EtgJGXSYe5Ha-JSW_ZC-6VURWKV4j1uu33GuLmyk/edit?usp=sharing |
| Kết quả lớn, checkpoint (Google Drive) | https://drive.google.com/drive/folders/1idI__VM_WHfRRVCFvXHccJCs3LhONktB?usp=sharing |
| Video demo | (bổ sung đường dẫn sau khi quay xong) |

### Danh sách thành viên và đóng góp

| Họ tên | MSSV | Nhiệm vụ và công việc đã thực hiện | Tỷ lệ đóng góp |
|---|---|---|---|
| Châu Hải Đăng | 3123410075 | Toàn bộ đồ án: engine, các tác tử, huấn luyện RL, arena và thí nghiệm, giao diện, kiểm thử, phân tích, báo cáo, slide, poster | 100% |

Tổng tỷ lệ đóng góp của nhóm: **100%**.

## Mục lục

1. [Đề tài](#1-đề-tài)
2. [Cài đặt](#2-cài-đặt)
3. [Cách chạy](#3-cách-chạy)
4. [Tái lập các thí nghiệm](#4-tái-lập-các-thí-nghiệm)
5. [Kết quả chính](#5-kết-quả-chính)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Dữ liệu và tài nguyên ngoài repo](#7-dữ-liệu-và-tài-nguyên-ngoài-repo)
8. [Môi trường đã dùng](#8-môi-trường-đã-dùng)
9. [Phần kế thừa và khai báo dùng công cụ AI](#9-phần-kế-thừa-và-khai-báo-dùng-công-cụ-ai)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

## 1. Đề tài

Trò chơi: Connect Four bàn chuẩn 7 cột × 6 hàng, hai người đi luân phiên, thả quân vào một cột, thắng khi có 4 quân liên tiếp (ngang, dọc, chéo), hòa khi bàn đầy.

| Cấp | Tác tử | Mô tả |
|---|---|---|
| 1 | Random | Chọn ngẫu nhiên đều trong các nước hợp lệ; dùng làm mốc so sánh. |
| 2 | Heuristic | Luật ưu tiên: thắng ngay, chặn nước thắng của đối thủ, tránh nước tạo cơ hội cho đối thủ, ưu tiên cột giữa. |
| 3 | Alpha-Beta | Negamax có cắt tỉa, giới hạn theo độ sâu hoặc thời gian (iterative deepening); có công tắc tắt cắt tỉa để so sánh. |
| 4 | MCTS (UCT) | Tìm kiếm cây Monte Carlo, giới hạn theo số mô phỏng hoặc thời gian. |
| 6 | Reinforcement Learning | Mạng Double DQN học bằng tự chơi (self-play). |

Cấp 5, 7, 8 của đề không được thực hiện (đã nêu trong phạm vi của báo cáo).

Hai câu hỏi nghiên cứu:

- **RQ1:** cùng thời gian suy nghĩ mỗi nước, MCTS có thắng nhiều hơn Alpha-Beta không? Kết quả có đổi khi tăng thời gian không?
- **RQ2:** tác tử RL mạnh tới đâu so với các tác tử khác, xét cả kết quả thi đấu và thời gian ra quyết định?

## 2. Cài đặt

Yêu cầu: Windows 11, **Python 3.10** (đã kiểm tra với 3.10.11). Có GPU NVIDIA là tùy chọn, chỉ cần khi huấn luyện RL.

### Công cụ cần có (tải ở đây)

| Công cụ | Dùng để | Tải về |
|---|---|---|
| Python 3.10 (khuyến nghị 3.10.11) | chạy toàn bộ đồ án | Tải thẳng bộ cài Windows 64-bit: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe (hoặc trang tải: https://www.python.org/downloads/windows/ , chọn Python 3.10.11, "Windows installer (64-bit)") |
| Visual Studio Code | mở dự án, chạy notebook, dùng terminal | https://code.visualstudio.com/download |

Khi cài Python, tick ô **Add python.exe to PATH**. Trong VS Code, cài hai tiện ích **Python** và **Jupyter** (mở thẻ Extensions, tìm tên và bấm Install) để chạy được notebook.

**Mở dự án bằng VS Code:** chọn *File → Open Folder* và mở thư mục gốc của dự án (nơi có `requirements.txt`), rồi mở terminal bằng *Terminal → New Terminal*; các lệnh bên dưới chạy trong terminal này. Để chạy notebook: mở file `.ipynb`, bấm *Select Kernel* ở góc phải phía trên và chọn môi trường `.venv` sau khi đã cài xong ở các bước dưới.

Chạy trong thư mục gốc của dự án, bằng PowerShell.

**Bước 1. Lấy mã nguồn**

```
git clone https://github.com/chauhaidang1307/TTNTNC_do_an_nhom17.git
cd TTNTNC_do_an_nhom17
```

Nếu nhận mã từ file ZIP thì giải nén rồi `cd` vào thư mục gốc (nơi có `requirements.txt`).

**Bước 2. Tạo và kích hoạt môi trường ảo**

```
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

Nếu PowerShell báo lỗi **"running scripts is disabled on this system"** (chính sách chạy script), mở khóa **chỉ cho cửa sổ terminal hiện tại** rồi kích hoạt lại:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Cách khác không cần đổi chính sách: dùng Command Prompt (cmd) và chạy `.venv\Scripts\activate.bat`, hoặc bỏ qua bước kích hoạt và gọi Python của môi trường ảo trực tiếp, ví dụ `.venv\Scripts\python.exe -m pip install -r requirements.txt` (ở các bước sau thay `python` bằng `.venv\Scripts\python.exe`). Khi kích hoạt thành công, đầu dòng lệnh có chữ `(.venv)`.

**Bước 3. Cài PyTorch trước, riêng một lệnh** (cài từ `requirements.txt` sẽ ra bản CPU)

Máy có GPU NVIDIA:

```
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
```

Máy không có GPU, hoặc driver không hỗ trợ CUDA 13.2:

```
python -m pip install torch
```

Với máy khác cấu hình, lấy lệnh phù hợp ở trang Get Started của pytorch.org.

Nếu lệnh cho GPU báo lỗi **"pip subprocess to install build dependencies did not run successfully"** hoặc **"Could not find a version that satisfies the requirement flit_core"**: nguyên nhân là lệnh trên chỉ cho pip tìm ở kho của PyTorch, trong khi một thư viện phụ thuộc của torch không có sẵn ở đó. Làm lần lượt:

1. Kiểm tra đúng Python 3.10 trong môi trường ảo: `python --version` (phải ra `Python 3.10.x`). Nếu không, xóa thư mục `.venv` và tạo lại ở Bước 2.
2. Cập nhật pip: `python -m pip install --upgrade pip`
3. Chạy lại, thêm kho PyPI để lấy các thư viện phụ thuộc:

```
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple
```

Nếu vẫn lỗi, tạm cài bản CPU (`python -m pip install torch`): mọi chức năng trừ việc huấn luyện RL nhanh đều dùng được.

**Bước 4. Cài các thư viện còn lại**

```
python -m pip install -r requirements.txt
```

**Bước 5. Kiểm tra**

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest -q
```

- Trên máy có GPU, kết quả đúng của dòng đầu là `2.14.1+cu132 13.2 True`. Nếu thấy `+cpu` hoặc `False` thì đang dùng bản CPU: các tác tử tìm kiếm, test và giao diện vẫn chạy bình thường, chỉ việc huấn luyện RL sẽ chậm hơn nhiều.
- Lệnh `pytest` cần ra **227 passed** (khoảng 35 giây trên máy tác giả).

**Cách khác:** mở [setup_environment.ipynb](setup_environment.ipynb) bằng Jupyter hoặc VS Code và chạy lần lượt từng ô; notebook làm đúng các bước 2 đến 5 và có giải thích từng ô.

Trên macOS/Linux: thay `py -3.10 -m venv .venv` bằng `python3.10 -m venv .venv`, kích hoạt bằng `source .venv/bin/activate`, các lệnh còn lại giữ nguyên. Đồ án được phát triển và kiểm tra trên Windows; các thành phần khác chưa được thử.

## 3. Cách chạy

Mọi lệnh chạy **từ thư mục gốc** (nơi có `requirements.txt`) với `.venv` đã kích hoạt.

### 3.1. Chạy kiểm thử

```
python -m pytest -q
```

### 3.2. Demo: chơi và quan sát tác tử

Giao diện web (Gradio) có ba chế độ: Người–Máy, Người–Người, Máy–Máy.

```
python -m ui.gui
```

Trình duyệt tự mở tại `http://127.0.0.1:7860`. Thêm `--no-browser` nếu không muốn tự mở, `--port 8000` để đổi cổng.

- Các tác tử Random, Heuristic, Alpha-Beta (chọn độ sâu hoặc thời gian) và MCTS dùng được ngay.
- Tác tử **Double DQN chỉ xuất hiện khi có file checkpoint** ở `checkpoints/<run>/last.pt` (xem mục 7). Checkpoint không nằm trên GitHub vì dung lượng.

Giao diện dòng lệnh, xem hai tác tử đấu với nhau trong terminal:

```
python -m ui.cli --p1 alphabeta --p2 heuristic --depth 4
```

Các tùy chọn: `--p1`, `--p2` chọn `random`, `heuristic` hoặc `alphabeta`; `--depth` độ sâu; `--time` giới hạn giây mỗi nước; `--no-pruning` tắt cắt tỉa; `--seed1`, `--seed2` cố định seed; `--delay` giây nghỉ giữa các nước. Cuối ván chương trình hỏi có lưu seed không (trả lời `y` hoặc `n`).

### 3.3. Đánh giá: chạy giải đấu

Giải đấu round-robin, mỗi cặp 100 ván (có thể đổi bằng `--games`, phải là số chẵn), chia đều đi trước và đi sau:

```
python -m arena.tournament --exp EXP-DEMO --split dev --games 10 --agents random heuristic ab-d3
```

Lệnh trên chạy thử nhỏ, mất vài giây. Log được ghi ở `logs/EXP-DEMO/<cặp>/` gồm `*_games.csv` (từng ván), `*_moves.csv` (từng nước: thời gian, số nút, độ sâu) và `*_meta.json` (cấu hình, seed, tóm tắt, môi trường).

Tên tác tử trong `--agents`:

| Tên | Ý nghĩa |
|---|---|
| `random`, `heuristic` | Cấp 1 và cấp 2 |
| `ab-d<N>` | Alpha-Beta độ sâu N, ví dụ `ab-d4` |
| `ab-t<ms>` | Alpha-Beta giới hạn thời gian, ví dụ `ab-t50` là 50 ms mỗi nước |
| hậu tố `-nopr` | tắt cắt tỉa (minimax thuần), ví dụ `ab-d4-nopr` |
| `mcts-s<N>` | MCTS với N mô phỏng, ví dụ `mcts-s1000` |
| `mcts-t<ms>` | MCTS giới hạn thời gian, ví dụ `mcts-t50` |
| `dqn_<run>_<checkpoint>` | Double DQN đã huấn luyện, ví dụ `dqn_RL-004_last` (cần file checkpoint) |

Lưu ý:
- Giải đấu **không ghi đè log cũ**: nếu mã `--exp` đã tồn tại thì lệnh dừng và yêu cầu mã mới. Để chạy lại một thí nghiệm có sẵn, dùng mã khác (ví dụ `EXP-009-B-LAI`).
- Dải seed **dev** bắt đầu từ 0, dải **test** từ 1.000.000; tác tử thứ hai của mỗi ván dùng seed cộng thêm 10.000.000. Dải test (`--split test`) bắt buộc thêm `--confirm-test` và chỉ nên chạy một lần sau khi chốt cấu hình; log test của đồ án đã có sẵn trong `logs/`.
- Với tác tử giới hạn thời gian (`ab-t`, `mcts-t`), kết quả phụ thuộc tốc độ máy, nên chạy lại ở máy khác có thể cho số khác.

Đo RAM và VRAM từng tác tử:

```
python -m arena.resources --exp EXP-011-LAI --games 10
```

### 3.4. Huấn luyện RL

Chạy thử nhanh để kiểm tra pipeline (300 ván, mạng nhỏ, khoảng 15 giây; kết quả chưa có ý nghĩa về sức mạnh):

```
python -m rl.trainer --config configs/rl_smoke.json --run RL-DEMO
```

Huấn luyện đầy đủ như đồ án (lần chạy RL-004 dùng cấu hình này và mất 6.780 giây trên GPU; nên có GPU):

```
python -m rl.trainer --config configs/rl_v1_seed1.json --run RL-NEW --max-minutes 360
```

- `--config`: file cấu hình (kiến trúc mạng, siêu tham số, lịch 3 giai đoạn, seed). Có sẵn trong `configs/`: `rl_v1.json`, `rl_v1_seed1.json`, `rl_v2_continue.json` (tiếp tục huấn luyện từ checkpoint cũ), `rl_smoke.json` (chạy thử rất ngắn để kiểm tra pipeline).
- `--run`: tên lần chạy, **không được trùng** với lần đã có. Log ghi ở `logs/<run>/`, checkpoint ghi ở `checkpoints/<run>/`.
- `--max-minutes`: dừng sau số phút này (vẫn lưu checkpoint). Lần huấn luyện RL-004 của đồ án chạy 200.000 ván trong 6.780,5 giây trên RTX 3050 Laptop; CPU chậm hơn nhiều.

### 3.5. Sinh lại bảng và hình từ log

**Cách đơn giản nhất:** mở [analysis/phan_tich.ipynb](analysis/phan_tich.ipynb), chọn kernel `.venv`, bấm Run All (khoảng 15 giây). Notebook chỉ đọc log, không cần GPU hay checkpoint, có ô kiểm tra môi trường ở đầu và liệt kê file đã sinh ở cuối.

- Bảng ghi ở `analysis/tables/`, hình ghi ở `analysis/figures/notebook/`.
- Mọi con số lấy từ log, không có số nhập tay.

Các hình sơ đồ và phân tích dùng trong báo cáo:

```
python analysis/report_figures.py
python analysis/report_analysis.py
```

## 4. Tái lập các thí nghiệm

Mọi lệnh chạy từ thư mục gốc. Mã `EXP-...` là tên thư mục log tương ứng trong `logs/`. Muốn chạy lại thì **đổi mã** (log cũ không bị ghi đè).

| Thí nghiệm | Split | Lệnh |
|---|---|---|
| EXP-001: round-robin Random, Heuristic, Alpha-Beta d3 và d4 | dev | `python -m arena.tournament --exp EXP-001 --split dev --games 100 --agents random heuristic ab-d3 ab-d4` |
| EXP-003, -D4, -D5: bật/tắt cắt tỉa (độ sâu 3, 4, 5) | dev | `python -m arena.tournament --exp EXP-003 --split dev --games 100 --agents ab-d3 ab-d3-nopr` (D4 dùng `ab-d4`, D5 dùng `ab-d5`) |
| EXP-004: MCTS theo số mô phỏng | dev | `python -m arena.tournament --exp EXP-004 --split dev --games 100 --agents mcts-s200 mcts-s1000 mcts-s5000 ab-d4` |
| EXP-005-T50, -T200: RQ1 (dev) | dev | `python -m arena.tournament --exp EXP-005-T50 --split dev --games 100 --agents ab-t50 mcts-t50` (T200 dùng `ab-t200 mcts-t200`) |
| EXP-008: chọn checkpoint RL | dev | `python -m arena.tournament --exp EXP-008 --split dev --games 100 --agents dqn_RL-003_last dqn_RL-004_ep100000 dqn_RL-004_ep120000 dqn_RL-004_ep140000 dqn_RL-004_ep160000 dqn_RL-004_ep180000 dqn_RL-004_last ab-d3 ab-d4 mcts-s1000` |
| EXP-009-A: giải đấu chính, chế độ A (50 ms mỗi nước) | test | `python -m arena.tournament --exp EXP-009-A --split test --confirm-test --games 100 --agents random heuristic ab-t50 mcts-t50 dqn_RL-004_last` |
| EXP-009-B: giải đấu chính, chế độ B (cấu hình mặc định) | test | `python -m arena.tournament --exp EXP-009-B --split test --confirm-test --games 100 --agents random heuristic ab-d4 mcts-s1000 dqn_RL-004_last` |
| EXP-010: RQ1 ở 200 ms (đo bổ sung, quyết định sau khi đã thấy kết quả 50 ms) | test | `python -m arena.tournament --exp EXP-010 --split test --confirm-test --games 100 --agents ab-t200 mcts-t200` |
| EXP-011: RAM và VRAM | - | `python -m arena.resources --exp EXP-011 --games 10` |
| RL-004: huấn luyện chốt (seed 1) | - | `python -m rl.trainer --config configs/rl_v1_seed1.json --run RL-004 --max-minutes 360` |

EXP-006 và EXP-007 (dev, có tác tử RL của các lần huấn luyện RL-002 và RL-003) cùng RL-001 đến RL-003 có log và cấu hình trong `logs/`; danh sách tác tử và tham số của từng cặp nằm trong file `meta.json` tương ứng. Các lệnh có `dqn_...` cần file checkpoint (mục 7).

Mỗi thí nghiệm đã chạy đều có đủ log thô trong `logs/<EXP>/`. Bảng tổng hợp mọi lần chạy (mã, ngày, cấu hình, kết quả, vị trí log) nằm ở Google Sheets (đường dẫn ở đầu file này).

**Quy tắc chia dữ liệu:** dải dev dùng để chọn tham số và checkpoint; dải test dùng một lần cho kết quả cuối, sau khi đã chốt checkpoint. Quy tắc chọn checkpoint được đặt trước (xem notebook, mục 8 và báo cáo mục 4.8).

## 5. Kết quả chính

Các số dưới đây lấy từ log (xem `analysis/tables/` và báo cáo). Score = (thắng + 0,5 × hòa) / số ván, **không** phải tỷ lệ thắng thuần.

| Nội dung | Kết quả |
|---|---|
| Cắt tỉa Alpha-Beta (dev, độ sâu 3 / 4 / 5) | số nút giảm 1,89× / 2,68× / 6,49×; thời gian giảm 2,16× / 2,92× / 7,64× |
| MCTS vs Alpha-Beta d4 (dev, 200 / 1.000 / 5.000 mô phỏng) | score của MCTS 0,160 / 0,225 / 0,475 |
| RQ1: score của Alpha-Beta trước MCTS cùng thời gian | dev 50 ms: 0,795; dev 200 ms: 0,800; test 50 ms: 0,765; test 200 ms: 0,635 (khoảng Wilson 0,537 đến 0,723, chồng với khoảng của 50 ms) |
| RQ2: score của DQN (test, chế độ B) | 1,00 trước Random; 0,97 trước Heuristic; 1,00 trước Alpha-Beta d4; 0,72 trước MCTS 1.000 mô phỏng |
| Thời gian mỗi nước (test, chế độ B, trung bình) | DQN 0,40 ms (p95 0,66); Alpha-Beta d4 9,7 ms (p95 23,1); MCTS 1.000 mô phỏng 54,3 ms (p95 98,9) |

Điều cần thận trọng khi đọc:
- Alpha-Beta và RL thường chơi gần như xác định, nên nhiều ván lặp: trong 100 ván DQN đấu Alpha-Beta d4 chỉ có 32 chuỗi nước khác nhau. Khoảng Wilson vì vậy có thể quá lạc quan.
- Đây là sức mạnh thực nghiệm của đúng các cấu hình đã chạy, không phải chứng minh tối ưu. Connect Four đã được giải về lý thuyết (Allis, 1988).
- Thời gian đo trên laptop, tải máy chưa kiểm soát hoàn toàn; chỉ nên đọc là so sánh thô.
- Mức test 200 ms cho RQ1 được thêm sau khi đã xem kết quả test 50 ms.

## 6. Cấu trúc thư mục

```
Connect4/
├── engine/          board.py (bàn cờ, luật, undo/clone), encoding.py (mã hóa trạng thái cho mạng)
├── agents/          base, random_agent, heuristic, alphabeta, mcts, dqn_agent
├── rl/              network, replay, trainer, checkpoint
├── arena/           match, tournament, metrics, resources (giải đấu, ghi log; độc lập với giao diện)
├── ui/              session, gui (Gradio), cli (dòng lệnh)
├── tests/           pytest (227 test)
├── configs/         cấu hình huấn luyện RL (JSON)
├── logs/            log thô: logs/EXP-*/ (giải đấu), logs/RL-*/ (huấn luyện)
├── analysis/        phan_tich.ipynb, report_figures.py, report_analysis.py, tables/, figures/
├── Concepts/        ghi chú khái niệm của tác giả (không cần để chạy)
├── setup_environment.ipynb   dựng môi trường từng bước
├── requirements.txt
└── README.md
```

Thư mục `checkpoints/` (trọng số mô hình) không có trên GitHub, xem mục 7.

## 7. Dữ liệu và tài nguyên ngoài repo

- **Dataset:** đồ án **không dùng dataset ngoài**. Dữ liệu huấn luyện RL do chính các ván tự chơi sinh ra; dữ liệu đánh giá là log giải đấu trong `logs/`. Bộ UCI Connect-4 không được sử dụng.
- **Checkpoint (mô hình đã huấn luyện):** nằm ở `checkpoints/<run>/` (`best.pt`, `last.pt` và các mốc `ep<N>.pt`). Thư mục này bị `.gitignore`. Lấy từ file ZIP nộp bài hoặc từ Google Drive (đường dẫn ở đầu file này), rồi đặt đúng vào `checkpoints/RL-004/` để chạy được `dqn_RL-004_last`. Checkpoint đã chốt cho giải đấu cuối là `RL-004/last.pt`.
- **Không cần checkpoint** để: chạy test, demo các tác tử không phải DQN, chạy giải đấu không có `dqn_...`, và sinh lại bảng, hình bằng notebook.
- **Log thô:** có sẵn trong `logs/` (khoảng 26 MB).

## 8. Môi trường đã dùng

- Hệ điều hành: Windows 11.
- CPU: Intel Core i5 thế hệ 12; GPU: NVIDIA GeForce RTX 3050 Laptop 4 GB, driver 617.42.
- Python 3.10.11; PyTorch 2.14.1+cu132; các thư viện còn lại ghim trong `requirements.txt`.
- Thời gian ra quyết định đo trên laptop này; máy khác sẽ cho số khác.

## 9. Phần kế thừa và khai báo dùng công cụ AI

**Tự cài đặt:** toàn bộ engine, các tác tử, huấn luyện RL, arena, giao diện, kiểm thử và mã phân tích trong repo này. Không dùng engine Connect Four hay mã tác tử có sẵn của bên thứ ba; các thuật toán (negamax, Alpha-Beta, UCT, Double DQN) là phương pháp đã công bố, cài đặt lại từ mô tả.

**Thư viện kế thừa:** PyTorch (mạng nơ-ron, huấn luyện), NumPy, pandas, matplotlib (bảng, hình), Gradio (giao diện web), pytest (kiểm thử), psutil (đo RAM), ipykernel (chạy notebook) và thư viện chuẩn của Python.

**Công cụ AI:** đồ án có dùng Claude (qua Claude Code) để hướng dẫn khái niệm, nhận xét và hỗ trợ kỹ thuật. Một phần mã (kiểm thử, khung arena, MCTS, phần RL, giao diện Gradio, mã sinh hình, notebook phân tích) do công cụ AI viết theo yêu cầu rõ ràng của tác giả và được kiểm tra lại; lõi `engine/board.py`, `agents/random_agent.py`, `agents/heuristic.py` và `ui/cli.py` do tác giả tự viết. Công cụ AI không tạo ra số liệu thực nghiệm: mọi số liệu đọc từ log do chương trình sinh ra. Chi tiết từng lần dùng nằm ở file `NHAT_KY.md` (kèm trong file ZIP nộp bài) và Phụ lục C của báo cáo.

## 10. Tài liệu tham khảo

1. L. V. Allis, "A knowledge-based approach of Connect-Four. The game is solved: White wins," M.Sc. thesis, Vrije Universiteit, Amsterdam, 1988.
2. J. Tromp, "John's Connect Four Playground," https://tromp.github.io/c4/c4.html
3. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2021.
4. D. E. Knuth and R. W. Moore, "An analysis of alpha-beta pruning," *Artificial Intelligence*, vol. 6, no. 4, pp. 293–326, 1975.
5. L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo planning," ECML 2006, LNCS 4212, pp. 282–293.
6. C. B. Browne et al., "A survey of Monte Carlo tree search methods," *IEEE Trans. Comput. Intell. AI Games*, vol. 4, no. 1, pp. 1–43, 2012.
7. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
8. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.
9. H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," AAAI 2016, pp. 2094–2100.
10. M. Thill, P. Koch, and W. Konen, "Reinforcement learning with n-tuples on the game Connect-4," PPSN XII, LNCS 7491, pp. 184–194, 2012.
11. E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, pp. 209–212, 1927.
12. D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, vol. 362, pp. 1140–1144, 2018.
13. M. Liu, *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*, CRC Press, 2024.
