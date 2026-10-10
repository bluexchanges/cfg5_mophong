# Đưa mã nguồn lên server và preprocess CFG5 lại

## Bản GitHub chứa gì?

Bản GitHub chỉ có code, config, test, split/manifest/index nhỏ, thống kê train-only của lần chạy cũ và báo cáo lịch sử. Nó **không** có raw CFG5, processed NPY, checkpoint, môi trường ảo hay luật làm việc riêng. `git clone` một mình chưa thể preprocess hoặc train.

Mã ACT đã tích hợp ở `third_party/rh20t_act/` và `src/models/rh20t_act.py`; trainer riêng của dự án ở `src/training/pretrain.py`. Các lệnh bên dưới dừng sau bước kiểm dữ liệu; bước train nằm trong [hướng dẫn pretraining](PRETRAINING_VI.md). Chưa có run pretraining thật.

## 1. Chuẩn bị raw và môi trường

Trên server Linux, clone repository rồi cung cấp raw CFG5 bằng kênh riêng (tải từ nguồn RH20T hoặc chuyển bản raw bạn đang có). Không đẩy raw lên GitHub. Đặt các episode robot ở `RH20T_cfg5/training_data/task_XXXX/<episode>/` nếu dùng cây raw đã được tổ chức, hoặc ở `RH20T_cfg5/<episode>/` nếu dùng layout gốc. Mỗi episode phải có `metadata.json`, `transformed/tcp_base.npy`, `transformed/gripper.npy` và 8 thư mục camera với `color.mp4` + `timestamps.npy`. Không chỉ copy thư mục `data/` từ GitHub rồi mong có video.

Kiểm phiên bản Python và cài dependency dữ liệu trong môi trường ảo riêng. Các bản lock hiện được thử trên Windows CPython 3.12; nếu wheel Linux không có, giải quyết dependency trước khi chạy toàn bộ. PyTorch CUDA cần được chọn theo GPU/driver trên server, không cài bản `+cpu` trong `requirements-act-lock.txt` để train.

```bash
cd /path/to/your-clone
python3 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data-lock.txt
```

## 2. Tạo manifest hợp với raw trên server, giữ nguyên membership

`source_fingerprint` hiện có trong manifest cũ gồm **mtime nanosecond** của file nguồn. Giải nén/tải lại có thể đổi mtime dù byte dữ liệu không đổi. Do đó tạo manifest server mới; script này khảo sát raw và có thể mất thời gian. Không sửa tay hash để vượt kiểm tra.

```bash
python scripts/prepare_manifest.py \
  --config configs/data/cfg5_all8.json \
  --output data/splits/cfg5_server_manifest.json
python scripts/create_splits.py \
  --config configs/data/cfg5_all8.json \
  --manifest data/splits/cfg5_server_manifest.json \
  --output data/splits/cfg5_server_splits.json
```

Trước khi decode ảnh, **so chính xác ba danh sách episode** với split đã khóa. Lệnh phải in `OK` và các số 531/66/66; nếu không, dừng để tìm khác biệt raw/config, không chọn lại seed theo validation/test.

```bash
python - <<'PY'
import json
from pathlib import Path
old = json.loads(Path('data/splits/cfg5_all_splits.json').read_text(encoding='utf-8'))
new = json.loads(Path('data/splits/cfg5_server_splits.json').read_text(encoding='utf-8'))
for name, count in [('train', 531), ('val', 66), ('test', 66)]:
    assert len(new['splits'][name]) == count, (name, len(new['splits'][name]))
    assert new['splits'][name] == old['splits'][name], f'{name} khác split đã khóa'
print('OK: train=531, val=66, test=66; membership không đổi')
PY
```

Manifest và split server có hash mới do fingerprint nguồn thay đổi; điều này **không đồng nghĩa đổi thành viên train/val/test** nếu phép so trên đạt. Không dùng lại `cfg5_all_normalization.json` cũ với hash mới; tính lại trên train server ở bước 4. Nếu raw có byte khác bản cũ, phép so episode có thể vẫn đạt nhưng thống kê/ảnh có thể khác; giữ nguồn và phiên bản dataset để truy vết.

## 3. Smoke 1 episode, rồi preprocess train và validation

```bash
python scripts/preprocess_data.py \
  --config configs/data/cfg5_all8.json \
  --manifest data/splits/cfg5_server_manifest.json \
  --splits data/splits/cfg5_server_splits.json \
  --split train --limit-episodes 1 --max-steps 8 \
  --output-root data/processed/cfg5_server_smoke --resume

python scripts/preprocess_data.py \
  --config configs/data/cfg5_all8.json \
  --manifest data/splits/cfg5_server_manifest.json \
  --splits data/splits/cfg5_server_splits.json \
  --split train --output-root data/processed/cfg5_all --resume
python scripts/preprocess_data.py \
  --config configs/data/cfg5_all8.json \
  --manifest data/splits/cfg5_server_manifest.json \
  --splits data/splits/cfg5_server_splits.json \
  --split val --output-root data/processed/cfg5_all --resume
```

Chạy từng lệnh và đợi mã thoát 0 trước lệnh sau. Smoke root tách riêng. Bản processed train+val trước đây gần **400,33 GB**; chuẩn bị ổ nhanh và dung lượng dự phòng. `--resume` bỏ qua episode hoàn chỉnh sau kiểm metadata/hash/shape, nhưng sẽ dừng ở thư mục `.partial` để bạn xem lỗi. Test giữ khóa cho đến khi chọn xong checkpoint/config.

## 4. Fit train-only normalization và kiểm loader

```bash
python scripts/fit_normalization.py \
  --config configs/data/cfg5_all8.json \
  --splits data/splits/cfg5_server_splits.json \
  --processed-root data/processed/cfg5_all \
  --output data/splits/cfg5_server_normalization.json
python scripts/check_dataset.py \
  --config configs/data/cfg5_all8.json \
  --splits data/splits/cfg5_server_splits.json \
  --stats data/splits/cfg5_server_normalization.json \
  --processed-root data/processed/cfg5_all --split train \
  --output docs/audit/cfg5_server_train_check.json
python scripts/check_dataset.py \
  --config configs/data/cfg5_all8.json \
  --splits data/splits/cfg5_server_splits.json \
  --stats data/splits/cfg5_server_normalization.json \
  --processed-root data/processed/cfg5_all --split val \
  --output docs/audit/cfg5_server_val_check.json
```

Các lệnh `check_dataset.py` chỉ đọc vài sample để kiểm shape/mask, không đánh giá loss của ACT. Sau khi đủ train+val và fit normalization trên server, xem [PRETRAINING_VI.md](PRETRAINING_VI.md) để chạy smoke, pretraining, validation toàn tập và resume. Trainer đã được viết nhưng chưa kiểm full ACT optimizer step trên server GPU.
