# Mã nguồn dùng trong dự án

`src` chứa logic có thể import; `scripts` chứa chương trình gọi logic đó từ PowerShell. `__init__.py` đánh dấu package, không phải chương trình training.

| Nhánh | Vai trò | Đọc trước |
|---|---|---|
| [data/](data/README.md) | Đọc CFG5, đồng bộ thời gian, tạo manifest/split, decode RGB, chuẩn hóa, tạo sample/action chunk. | `cfg5.py`, rồi `act_dataset.py`. |
| [simulation/](simulation/README.md) | Tạo scene, gắn camera, điều khiển Panda bằng IK, render observation và đánh giá đặt3 chốt. | `panda_env.py`, rồi `panda_controller.py` và `panda_evaluator.py`. |
| [training/](training/pretrain.py) | Ghép ACTDataset với model, chạy optimizer/validation, lưu checkpoint và resume tại ranh giới epoch. | `pretrain.py`, rồi [hướng dẫn chạy](../docs/PRETRAINING_VI.md). |

Hai nhánh gặp nhau ở representation 8D: TCP aligned base xyz + quaternion wxyz + gripper width(m), cùng thứ tự8 camera. `simulation/panda_observation.py` đưa observation về shape/normalization của `ACTDataset`.

Luồng rollout dự kiến là ảnh+state → model → action chunk → chọn action cần thực thi → denormalize → kiểm action → IK/actuator → physics → evaluator. Hiện đã có phần dữ liệu, mô phỏng, [models/](models/README.md) và trainer CFG5. Trainer mới đã kiểm bằng fixture và ACT thu nhỏ, chưa có full ACT optimizer step trên CFG5 hoặc learned-policy rollout. `panda_demo.py` là controller có truy cập trạng thái vật thật trong simulator, không phải policy học từ ảnh.

Không import snapshot trong `references/upstream/` như dependency training. Khi đổi schema đầu vào/đầu ra, cập nhật cả config, adapter, tests và tài liệu. Xem [tests](../tests/README.md) và [scripts](../scripts/README.md).

Nhánh [models/](models/README.md) chứa adapter RH20T ACT dùng source đầy đủ ở third_party. Đã có forward/loss/optimizer factory; [training/](training/pretrain.py) điều phối optimizer step và đã kiểm với ACT thu nhỏ, chưa pretrain trên CFG5.
