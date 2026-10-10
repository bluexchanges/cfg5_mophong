# RH20T CFG5 → ACT → Franka Panda trong MuJoCo

**Cập nhật máy B, 2026-09-30:** Code trainer CFG5, validation, checkpoint/resume đã thêm vào `src/training/`, `scripts/train.py`, `scripts/validate.py`; đã kiểm bằng fixture và ACT thu nhỏ, chưa chạy full ACT optimizer step với CFG5. Bản processed train+val của máy A đã được dọn; metadata split/normalization lịch sử nằm trong `data/`, còn server sẽ preprocess và tính statistics lại. Xem [hướng dẫn pretraining](docs/PRETRAINING_VI.md) và [preprocess server](docs/SERVER_REPROCESS_VI.md). Chưa có checkpoint ACT được học.

Dùng RH20T CFG5 để **pretraining ACT**, sau đó fine-tune cho Panda **gắp và đặt vật đúng vào khuôn nhỏ** trong MuJoCo với 8 camera theo calibration thật. Hướng dẫn tiếng Việt; training dài do người dùng chạy.

**Trạng thái:** PandaEnv/MuJoCo,8 camera (6 cố định +2 in-hand), controller và evaluator cho **3 chốt Ø6×35mm vào3 lỗ Ø7mm, sâu12mm**. Đã sửa logic đo độ nhô23mm±0.5mm, support đáy và các điều kiện chống false success; xem [báo cáo sửa placement](docs/PLACEMENT_LOGIC_REVIEW_VI.md). Controller sau sửa đặt đủ3 vật ở seeds42/43/44. Viewer/headless đã chạy; ACT baseline đã tải và tích hợp, full inference trên một sample train smoke và backward nhỏ đạt; chưa optimizer/pretraining/ACT rollout. Xem [tích hợp ACT](docs/ACT_SETUP_VI.md) và [hướng dẫn tự chọn/chuyển/xem CFG5](docs/CFG5_HANDS_ON_VI.md).

Representation đã chọn: TCP aligned base xyz + quaternion wxyz + gripper width mét, 8D. **Toàn CFG5** đã được lọc:663 robot episode đạt ACT trong92/104 task;1.194 human video được giữ cho nghiên cứu one-shot và có chỉ mục liên hệ human–robot. Split full ngẫu nhiên seed42 là531 train/66 val/66 test. Trên máy A, người dùng từng preprocess **đủ531 train+66 val** rồi kiểm audit/normalization; output lớn đã dọn, cần tạo lại trên server. Test còn khóa. ACT chưa train. Target pose là future measured-state proxy; task khuôn cần demonstrations riêng để fine-tune. Xem [báo cáo lọc](docs/CFG5_FULL_FILTER_VI.md), [audit lịch sử](docs/audit/cfg5_all_processed_postrun.json) và [hướng dẫn server](docs/SERVER_REPROCESS_VI.md).

## Chạy ngay

```powershell
Set-Location 'D:\Download\mophong_RH20T'
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/run_pick_place_demo.py --viewer --video --output outputs/my_demo
```

Chạy bộ tests dữ liệu/mô phỏng/placement bằng đúng `.venv-data` (Windows CPython3.12.14); Python trên PATH của máy A là MSYS2. Xem [hướng dẫn MuJoCo/Panda](docs/MUJOCO_PANDA_GUIDE_VI.md). Kết quả evaluator: `outputs/placement_logic_review/`. Video lượt3 trước sửa: `outputs/three_pegs_demo/controller_demo.mp4`; đó là scripted controller, chưa phải ACT. Output processed 531 train+66 validation của máy A đã dọn sau khi kiểm; 66 test episode chưa xử lý. Chưa train ACT.

## Bắt đầu từ đây

1. [Báo cáo khảo sát](docs/PROJECT_AUDIT.md): máy hiện có gì, đã kiểm tra gì và giới hạn.
2. [Phân tích paper](docs/PAPER_ANALYSIS.md): ACT và RH20T thực sự làm gì.
3. [So sánh implementation](docs/ACT_IMPLEMENTATION_REVIEW.md): ba repo ACT, validation density và các lỗi đã phát hiện.
4. [Quyết định mới nhất](docs/DECISIONS.md), [hướng dẫn dữ liệu](docs/DATA_GUIDE_VI.md), [setup8 camera](docs/CAMERA_SETUP_VI.md).
5. [Protocol pretraining và đặt vào khuôn](docs/EXPERIMENT_PROTOCOL.md), [báo cáo lượt2](docs/STAGE2_REPORT_VI.md).
6. [Kho tài liệu tham khảo](references/README.md): bỏ thêm PDF vào `references/papers/`, ghi chú vào `references/notes/`.

## Chạy lại kiểm kê dữ liệu

Mở PowerShell, chạy từ project root:

```powershell
Set-Location 'D:\Download\mophong_RH20T'
python --version
python scripts/inspect_data.py --scan-files
```

Input: `RH20T_cfg5/`. Output: `docs/audit/project_inventory.json`. Script mặc định chỉ dùng Python standard library (Python >= 3.10); không cần cài PyTorch/MuJoCo. Lần kiểm tra hiện tại dùng Python 3.12.7.

Kết quả kiểm kê ngày 2026-09-27: 2.512 thư mục đúng mẫu tên gồm 1.321 robot và 1.191 human, cộng 3 thư mục tên kết thúc `_human_2`; 104 task ID. Những số này không xác nhận mọi episode đủ dữ liệu hoặc dùng được để train.

Để xem object NPY hoặc chạy probe sampling cần NumPy. Lượt1 dùng runtime có sẵn; lượt2 đã tạo `.venv-data` riêng với NumPy/OpenCV, khóa phiên bản tại `requirements-data-lock.txt`. [PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md) là báo cáo lịch sử lượt1.

Lỗi thường gặp: chạy sai thư mục làm không tìm thấy `RH20T_cfg5`; dùng `--root` để chỉ đường dẫn khác. Không đặt `--output` trong dữ liệu gốc: script chủ động từ chối.

## Cấu trúc hiện tại

Mỗi thư mục trong bảng có README giải thích các file, luồng xử lý và điểm cần chú ý. Bắt đầu ở bảng này khi chưa biết phần nào chịu trách nhiệm cho một chức năng.

| Thư mục | Hướng dẫn |
|---|---|
| [configs](configs/README.md) | Tham số dữ liệu/task/camera, alias và quan hệ giữa các config. |
| [data](data/README.md) | Manifest/split/statistics, từng array processed và phân biệt smoke/full. |
| [third_party](third_party/README.md) | Source ACT đầy đủ dùng trực tiếp, commit/hash/license. |
| [src](src/README.md) | Bản đồ code; chi tiết từng module ở [data](src/data/README.md) và [simulation](src/simulation/README.md). |
| [scripts](scripts/README.md) | Từng lệnh chạy, input/output, thứ tự pipeline và giới hạn hiện tại. |
| [tests](tests/README.md) | Các lỗi mỗi nhóm test kiểm, cách chạy và phạm vi kết luận. |
| [mujoco](mujoco/README.md) | Model/mesh upstream, scene được sinh và manifest nguồn. |
| [outputs](outputs/README.md) | Đọc result/trace/CSV/video; kết quả placement mới và các run lịch sử. |
| [docs](docs/README.md) | Thứ tự đọc tài liệu; [audit](docs/audit/README.md) mô tả từng bằng chứng kiểm thử. |
| [references](references/README.md) | PDF, text trích xuất, ghi chú, yêu cầu ban đầu và snapshot source. |
| [.venv-data](.venv-data/README.md) | Windows Python đang dùng, packages và lock files. |
| [.venv](.venv/README.md) | Môi trường MSYS2 cũ và cách tránh chọn nhầm interpreter. |
| [tmp](tmp/README.md) | Ảnh trang PDF/file kiểm tra tạm, không phải nguồn chuẩn. |

Payload trong `RH20T_cfg5/` vẫn nguyên vẹn. Theo yêu cầu mới, toàn bộ episode đã được move theo task vào `training_data/` và `missing_data/`; CSV metadata nhỏ tại `human_robot_index.csv` giữ liên hệ theo tên, không phải nhãn framewise. Schema ở [DATA_GUIDE_VI](docs/DATA_GUIDE_VI.md) và số liệu move ở [CFG5_FULL_FILTER_VI](docs/CFG5_FULL_FILTER_VI.md).

```text
mophong_RH20T/
├── RH20T_cfg5/             # Payload gốc, episode đã move theo task; CSV chỉ mục nhỏ
├── references/             # Paper, ghi chú, nguồn và snapshot khảo sát
├── docs/                   # Phân tích và báo cáo; audit/ chứa bằng chứng nhỏ
├── configs/                # Dữ liệu, camera rig và thiết kế task khuôn
├── data/splits/            # Manifest, split, smoke statistics
├── data/processed/         # Output xử lý lại trên server; không đưa vào Git
├── third_party/            # Source RH20T ACT baseline nguyên vẹn để import
├── src/models/             # Adapter ACT8 camera/8D và masked loss
├── src/data/               # Timestamp, preprocessing, normalization, ACTDataset
├── src/simulation/         # PandaEnv, IK/controller, evaluator, camera và ACT observation adapter
├── mujoco/                 # Upstream Panda, generated scene và asset download manifest
├── scripts/                # Inventory, preprocessing, split, checks
├── tests/                  # Unit tests với fixture nhỏ
├── AGENTS.md
├── requirements.txt        # Dependency dữ liệu và mô phỏng đã khóa phiên bản
└── .gitignore
```

Thứ tự hiện tại: đưa bản code sạch lên server → preprocess lại CFG5 train+val → smoke full ACT trên server GPU → bạn chạy pretraining → demonstrations khuôn/fine-tuning → đánh giá checkpoint, theo [DECISIONS.md](docs/DECISIONS.md). Thư mục chưa phải Git repository; chưa tạo commit.
