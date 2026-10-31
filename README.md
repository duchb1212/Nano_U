# Nano-U — Tái hiện thực nghiệm trên MacBook Air M5

> **Trạng thái:** Đã tái hiện thành công bước **đánh giá (evaluation)** mô hình Nano-U INT8 pretrained trên TinyAgri **Test (181 ảnh)** và **Validation (362 ảnh)**. **Chưa huấn luyện lại từ đầu** và **chưa xác nhận demo chạy từng ảnh riêng lẻ thành công**.

## 1. Tổng quan dự án

Nano-U là mô hình phân đoạn ngữ nghĩa nhị phân (binary terrain segmentation) nhỏ gọn, xác định vùng địa hình robot có thể di chuyển dựa trên ảnh RGB. Mô hình sử dụng kiến trúc encoder–decoder và được thiết kế hướng tới triển khai trên phần cứng nhúng có tài nguyên hạn chế.

- **Mã nguồn gốc:** https://github.com/federico-pizz/Nano-U
- **Bộ dữ liệu:** https://huggingface.co/datasets/federico-pizz/TinyAgri
- **Phạm vi tài liệu:** Tái hiện quá trình cài đặt, chuẩn bị dữ liệu và đánh giá mô hình INT8 có sẵn trên macOS.
- **Không tuyên bố:** Đã train lại Nano-U, train BU-Net Teacher, tái hiện toàn bộ QAD hoặc đo hiệu năng trên ESP32-S3.

## 2. Môi trường thực nghiệm

| Thành phần | Thông tin |
|---|---|
| Máy | MacBook Air M5, Apple Silicon (`arm64`) |
| RAM / SSD | 16 GB / 512 GB |
| Hệ điều hành | macOS |
| Python cho dự án | 3.12.x trong `.venv` (VS Code từng hiển thị 3.12.16) |
| TensorFlow | 2.20.0 (đã import thành công) |
| Gói Nano-U | `nanou==2.0.0` (theo đầu ra cài đặt) |
| Git | Apple Git 2.50.1 |
| Trình soạn thảo | VS Code và Terminal |

**Ghi nhận phiên bản source để đảm bảo tái lập:**

```bash
git rev-parse HEAD
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
python -m pip check
```

Ban đầu repository ở commit `c1f3a7ca18268d1b8d3b6fb1f050ef8914d69e2d`, sau đó đã chạy cập nhật lên cấu trúc script mới. **Commit hiện tại cần được lấy lại bằng lệnh ở trên**, không mặc định là commit ban đầu.

## 3. Các bước đã thực hiện

### 3.1. Clone và cập nhật repository

```bash
mkdir -p ~/VSCode
cd ~/VSCode
git clone https://github.com/federico-pizz/Nano-U.git
cd Nano-U

# Đã cập nhật từ cấu trúc phiên bản cũ lên phiên bản mới
git pull --ff-only
```

Ở bản cũ, mô hình mang tên `nano_u.tflite`; sau khi cập nhật, tên file trong repository là `models/TinyAgri/nanou.tflite`. Các script mới nằm trong `scripts/get_data.py`, `scripts/evaluate.py`, `scripts/train.py`.

### 3.2. Tạo môi trường Python riêng và cài dependencies

Mac có Python 3.14.7 nhưng dự án dùng Python 3.12 để tương thích thư viện:

```bash
brew install python@3.12
cd ~/VSCode/Nano-U
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

**Đã xác nhận:** `No broken requirements found`; `tensorflow.__version__ == 2.20.0`.

### 3.3. Tải và kiểm tra dữ liệu TinyAgri

```bash
python scripts/get_data.py --dry-run
python scripts/get_data.py --workers 4
python scripts/get_data.py --verify
```

Kết quả thực tế của bước verify:

```text
.../data/TinyAgri: train 1267, val 362, test 181
```

| Tập | Số cặp ảnh–mask | Vai trò |
|---|---:|---|
| Train | 1.267 | Dùng cho huấn luyện về sau; chưa dùng để train trong lần tái hiện này |
| Validation | 362 | Đánh giá trên tập validation |
| Test | 181 | Đánh giá mô hình pretrained trên tập test |
| **Tổng** | **1.810** | **3.620 file ảnh và mask** |

Dữ liệu đặt tại `data/TinyAgri/{train,val,test}/{img,mask}`. Ảnh được ghép với nhãn Ground Truth tương ứng để tính metric; mô hình nhận ảnh và tạo mask dự đoán. Nguồn dữ liệu là TinyAgri do tác giả phát hành, **không phải dữ liệu tự thu thập**.

### 3.4. Đánh giá mô hình INT8 đã huấn luyện sẵn

**Trên tập test:**

```bash
python scripts/evaluate.py nanou --preset tinyagri --format int8
```

**Trên tập validation:**

```bash
python scripts/evaluate.py nanou --preset tinyagri --split val --format int8
```

Cả hai lần đánh giá đã tạo báo cáo và hình ảnh đầu ra. Ngưỡng phân đoạn mặc định là **0,5**.

## 4. Kết quả đã thu được

### 4.1. Các chỉ số tổng hợp

| Chỉ số | Test (181 ảnh) | Validation (362 ảnh) |
|---|---:|---:|
| mIoU | **89,3031%** | **89,95%** |
| F1 | **94,2706%** | **94,80%** |
| Dice | **94,2706%** | **94,80%** |
| Precision | **93,3775%** | **93,99%** |
| Recall | **95,1810%** | **95,61%** |
| F0.5 | **93,7327%** | **94,31%** |
| F2 | **94,8147%** | **95,29%** |
| BCE | 0,157269 | 0,1428 |
| Focal loss | 0,062394 | 0,0511 |

**Độ chính xác số liệu:** Cột Test lấy từ JSON do chương trình xuất; cột Validation lấy từ kết quả Terminal được làm tròn đến 4 chữ số thập phân. Giá trị có thể được kiểm tra lại trong `eval_int8_val.json`.

Theo README upstream ở thời điểm đối chiếu, kết quả tác giả làm tròn trên TinyAgri Test cho Nano-U INT8 là **mIoU 89,3%, F1 94,3%, Precision 93,4%, Recall 95,2%**. Kết quả Test tại máy **khớp với mức làm tròn** này, cho thấy đã tái hiện phép đánh giá trên mô hình phát hành.

### 4.2. Kết quả theo từng chuỗi cảnh (Test)

| Nhóm ảnh | mIoU | F0.5 |
|---|---:|---:|
| `Crops_scene1` | 82,88% | 86,31% |
| `Crops_scene2` | 81,44% | 86,93% |
| `Tomatoes_scene1` | 93,29% | 97,19% |
| `Tomatoes_scene2` | 92,85% | 95,77% |

**Nhận xét:** Các cảnh Tomatoes đạt kết quả cao hơn các cảnh Crops trên tập test này. Chỉ từ các chỉ số chưa thể kết luận nguyên nhân; cần kiểm tra ảnh và mask cụ thể để xác định tác động của cây trồng, ánh sáng, bóng đổ và cấu trúc đường đi.

### 4.3. Khảo sát ngưỡng phân đoạn

Script đánh giá có `threshold_sweep` từ **0,05 đến 0,95**. Một vài kết quả từ Test:

| Threshold | Precision | Recall | F1 | mIoU |
|---|---:|---:|---:|---:|
| 0,35 | 92,37% | 96,59% | 94,43% | 89,46% |
| 0,50 | 93,38% | 95,18% | 94,27% | 89,30% |
| 0,80 | 95,43% | 90,21% | 92,75% | 87,08% |

**Nhận xét:** Threshold tăng làm Precision tăng và Recall giảm. Threshold 0,35 có mIoU và F1 cao nhất trong các ngưỡng quét **trên Test**, nhưng **không được dùng tập Test để chọn threshold rồi coi đó là một cải tiến khách quan**. Muốn thử tối ưu threshold cần chọn trên Validation trước, sau đó xác nhận trên Test.

## 5. Các file và thư mục — nguồn gốc, mục đích, trạng thái

```text
Nano-U/
├── .venv/                         # TẠO Ở MÁY: môi trường Python riêng
├── config/
│   ├── base.yaml                  # MÃ NGUỒN GỐC: cấu hình mặc định
│   └── presets/                   # MÃ NGUỒN GỐC: các preset cho dataset
├── data/
│   └── TinyAgri/                  # TẢI VỀ: dữ liệu ảnh và mask
│       ├── train/{img,mask}/
│       ├── val/{img,mask}/
│       └── test/{img,mask}/
├── models/
│   └── TinyAgri/
│       └── nanou.tflite           # MODEL PRETRAINED DO TÁC GIẢ CUNG CẤP
├── scripts/
│   ├── get_data.py                # MÃ NGUỒN GỐC: tải / kiểm tra dữ liệu
│   ├── evaluate.py                # MÃ NGUỒN GỐC: đánh giá, xuất JSON và hình
│   └── train.py                   # MÃ NGUỒN GỐC: huấn luyện (chưa chạy)
├── results/
│   └── TinyAgri/nanou/           # SINH RA KHI ĐÁNH GIÁ
│       ├── eval_int8_test.json    # Metrics Test + threshold sweep + per_sequence
│       ├── eval_int8_test.png     # Hình minh họa đánh giá Test
│       ├── eval_int8_test_metrics.png  # Biểu đồ metrics Test
│       ├── eval_int8_val.json     # Metrics Validation
│       ├── eval_int8_val.png      # Hình minh họa đánh giá Validation
│       ├── eval_int8_val_metrics.png   # Biểu đồ metrics Validation
│       └── baseline_05/          # TỰ TẠO: bản sao các file Test gốc
├── run_tinyagri.py               # TỰ THÊM: demo chọn ảnh bằng --split/--index
├── requirements.txt              # MÃ NGUỒN GỐC: các thư viện cần cài
└── README.md                     # README GỐC CỦA TÁC GIẢ (nên giữ nguyên)
```

### Phân biệt từng loại file

- **File upstream (không phải tự viết):** Các script trong `scripts/`, cấu hình, mã nguồn `src/`, `requirements.txt` và model pretrained.
- **File tạo/tải trên máy:** `.venv/`, `data/TinyAgri/` và `results/TinyAgri/nanou/eval_*`.
- **File/thư mục tự thêm:** `run_tinyagri.py` và thư mục `results/TinyAgri/nanou/baseline_05/`.
- **`run_tinyagri.py`:** Mục tiêu là lựa chọn ảnh theo `--split` và `--index`, xem Input RGB, Ground Truth, bản đồ Probability, Prediction và IoU riêng một ảnh; **chưa xác nhận chạy thành công**. Ở lần kiểm tra gần nhất, thư mục `custom_samples/` **chưa xuất hiện**.
- **`baseline_05/`:** Sao lưu kết quả Test tại threshold 0,5 để tránh ghi đè khi thí nghiệm bổ sung.

## 6. Cách chạy lại từ môi trường đã cài

```bash
cd ~/VSCode/Nano-U
source .venv/bin/activate
python scripts/get_data.py --verify

# Đánh giá tập test
python scripts/evaluate.py nanou --preset tinyagri --format int8

# Đánh giá tập validation
python scripts/evaluate.py nanou --preset tinyagri --split val --format int8

# Mở thư mục báo cáo
open results/TinyAgri/nanou
```

Để xem chỉ số dưới dạng text:

```bash
python -m json.tool results/TinyAgri/nanou/eval_int8_test.json
python -m json.tool results/TinyAgri/nanou/eval_int8_val.json
```

**Script demo cá nhân (đang kiểm tra):**

```bash
python -u run_tinyagri.py --split test --index 0
```

Nếu demo hoạt động đúng, mục tiêu là lưu ảnh vào `results/TinyAgri/nanou/custom_samples/`. **Không ghi nhận thư mục này như kết quả đã có cho đến khi script in đường dẫn lưu và file tồn tại.**

## 7. Phạm vi đã đạt được và chưa đạt được

### Đã hoàn thành

- Clone và cập nhật mã nguồn, cài môi trường tương thích trên Mac M5.
- Tải đủ 1.810 cặp ảnh–mask và xác minh Train/Val/Test.
- Sử dụng `.tflite` INT8 tác giả cung cấp để đánh giá Test và Validation.
- Thu được JSON, hình phân đoạn minh họa và biểu đồ thống kê.
- Tái hiện các chỉ số Test của repository hiện tại với mức làm tròn tương đương kết quả tham chiếu.
- Phân tích sơ bộ Precision/Recall theo threshold và sự khác biệt giữa các chuỗi cảnh.

### Chưa thực hiện hoặc chưa xác minh

- Chưa xác minh demo `run_tinyagri.py` chạy và lưu ảnh đơn thành công.
- Chưa train Nano-U từ đầu trên TinyAgri hoặc dữ liệu khác.
- Chưa huấn luyện Teacher BU-Net và chưa tái hiện đầy đủ phương pháp QAD.
- Chưa đánh giá trên tập ảnh tự thu thập hoặc BotanicGarden.
- Chưa đo độ trễ, RAM, điện năng trên thiết bị ESP32-S3; các thông số phần cứng trong bài báo là **của tác giả**.

## 8. Hướng phát triển tiếp theo

1. Kiểm tra và sửa `run_tinyagri.py`, xuất kết quả cho từng ảnh tiêu biểu trong Crops/Tomatoes.
2. Thử threshold trên **Validation**, sau đó đo lại trên Test; ghi rõ cách chọn để tránh rò rỉ dữ liệu đánh giá.
3. Tạo preset riêng và đường dẫn model/output riêng để train Nano-U độc lập, không ghi đè model pretrained.
4. Nếu tài nguyên cho phép, nghiên cứu và tái hiện quy trình QAD gồm Teacher BU-Net, Student Nano-U và lượng tử hóa INT8.
5. Chuẩn bị dataset mới có ảnh và mask nhị phân để thử khả năng tổng quát hóa.

## 9. Tài liệu tham khảo

- [Nano-U — GitHub chính thức](https://github.com/federico-pizz/Nano-U)
- [TinyAgri — Hugging Face](https://huggingface.co/datasets/federico-pizz/TinyAgri)
- [Bài báo Nano-U](https://arxiv.org/abs/2605.10210)

> **Lưu ý về báo cáo khoa học:** Kết quả ở tài liệu này là từ **mô hình pretrained của phiên bản repository hiện tại**, không đồng nghĩa với tái hiện toàn bộ số liệu của phiên bản bài báo ban đầu. Cần lưu commit hash và preset chính xác trước khi trích dẫn, so sánh.
