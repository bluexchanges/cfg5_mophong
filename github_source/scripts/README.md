# Các chương trình chạy từ PowerShell

## Phân loại toàn CFG5 (2026-09-28)

| Script | Vai trò và giới hạn |
|---|---|
| `organize_cfg5.py` | `--plan` kiểm mọi episode; `--apply` move folder cùng ổ theo task với resume và đối chiếu count/byte/identity; `--verify` kiểm lại; `--rollback` trả về root nếu cần. Move đã hoàn tất, hiện chỉ cần `--verify`. Không copy video. |
| `manifest_from_organization.py` | Từ plan đã verify tạo `data/splits/cfg5_all_manifest.json` cho toàn104 task, không decode video lần hai. Manifest đã tạo, script không ghi đè. |
| `index_human_robot_cfg5.py` | Từ plan đã verify tạo CSV chung, CSV theo task và JSON chỉ mục human–robot. Ghép theo mã tên, không sinh action label human. Chỉ ghi metadata nhỏ, không move/copy episode. |

Các lệnh mặc định `cfg5_base8.json`/`cfg5_manifest.json` phía dưới thuộc subset lịch sử. Phạm vi mới dùng `configs/data/cfg5_all8.json`, `data/splits/cfg5_all_manifest.json` và `cfg5_all_splits.json` (ngẫu nhiên seed42, đúng531/66/66). Người dùng từng full preprocess531 train+66 val rồi dọn output; test/training chưa chạy. `preprocess_data.py --resume` kiểm/bỏ qua episode hoàn chỉnh khi chạy lại, không bỏ qua `.partial`. Trên server bắt đầu ở [hướng dẫn server](../docs/SERVER_REPROCESS_VI.md).

Chạy từ project root `D:\Download\mophong_RH20T` bằng `.venv-data\Scripts\python.exe -X utf8`. Các script là entrypoint; thuật toán chính nằm trong [src](../src/README.md). Các đường dẫn input/output dưới đây tính từ project root, là giá trị mặc định nếu không ghi khác.

## Khảo sát và pipeline dữ liệu

| File | Input → output và logic | Lưu ý |
|---|---|---|
| `_bootstrap.py` | Đưa project root vào đường dẫn import để script tìm `src`. | Helper, không phải bước người dùng cần chạy riêng. |
| `inspect_data.py` | `RH20T_cfg5` → `docs/audit/project_inventory.json`; kiểm tên episode, metadata, NPY header; `--scan-files` đếm file/dung lượng. | `--trusted-npy-samples` mới mở object payload với NumPy; không decode toàn bộ video. |
| `probe_validation_sampling.py` | Snapshot upstream+fixture timestamps → `docs/audit/upstream_sampling_probe.json`. Xác minh hash, trích hàm sampling rồi tái hiện hành vi padding/density. | Không import cả model hay chạy validation ACT. |
| `prepare_manifest.py` | `configs/data/cfg5_base8.json`+raw → `data/splits/cfg5_manifest.json`. Kiểm task/rating/8 camera/timeline và ghi lý do loại. | Có kiểm frame đầu; không chứng minh toàn bộ video tốt. |
| `create_splits.py` | Config+manifest → `data/splits/cfg5_splits.json`. Group split có seed và hash. | Thực hiện trước khi dùng action chunks, không chia observation ngẫu nhiên. |
| `preprocess_data.py` | Config+manifest+split+raw → `data/processed/<run>/<episode>/`. Decode RGB, lưu arrays/metadata. | Bắt buộc `--split train\|val\|test`; `--limit-episodes`/`--max-steps` phục vụ smoke. `--resume` kiểm và bỏ qua output hoàn chỉnh; `.partial` bị từ chối. |
| `fit_normalization.py` | Processed train+split → `data/splits/normalization.json`. Tính mean/std8D theo train. | `--allow-partial-smoke` tạo statistics smoke; không hợp thức hóa training thiếu dữ liệu. |
| `check_dataset.py` | Processed+split+stats → `docs/audit/dataset_check.json`. Lập ACTDataset, kiểm sample đầu/giữa/cuối và shape/mask. | Số sample trong report là độ dài dataset; số sample thực được kiểm nằm ở `checks`, không phải full evaluation. |
| `export_camera_rig.py` | Calibration mặc định1640071679624+config/API robot → `configs/mujoco/cfg5_camera_rig.json`. | Scale K, đổi frame và gắn parent cho6 fixed+2 wrist. |
| `audit_camera_rig.py` | Rig+raw calibration+split → `docs/audit/camera_rig_check.json`. Kiểm số học transform/roundtrip và đối chiếu TCP. | Tách biệt với kiểm projection bằng renderer. |

## Mô phỏng

| File | Input → output và logic | Lưu ý |
|---|---|---|
| `download_panda_model.py` | Tải riêng Panda Menagerie theo commit khóa → `mujoco/models/franka_emika_panda/`, kèm license và manifest hash. | Cần network nếu thiếu asset. Không phải script tải ACT/checkpoint. |
| `build_panda_scene.py` | Task config+rig+Panda model → `mujoco/scenes/panda_three_pegs.xml` và manifest. | Sửa config/builder rồi sinh lại; giữ upstream XML. |
| `check_mujoco.py` | Build/load scene, render8 camera, kiểm observation adapter và projection marker → `docs/audit/mujoco_check.json`, ảnh `outputs/stage3/`. | `--output` đổi report nhưng ảnh preview vẫn ghi ở `outputs/stage3/`; các file tại đó có thể được cập nhật. |
| `run_panda_viewer.py` | Mở PandaEnv trong viewer; `--seconds` đặt thời gian tự đóng. | Xem scene, không đồng nghĩa chạy hết demo pick-place. |
| `example_panda_env.py` | Ví dụ import/reset/step/close và kiểm observation. | Mẫu tối giản để viết chương trình của bạn. |
| `run_pick_place_demo.py` | Scripted controller, seed/config → `outputs/three_pegs_demo/`: result, trace, ảnh cuối; `--video` thêm MP4; `--viewer` mở GUI. | Có privileged state. Chọn `--output` mới để giữ kết quả lịch sử. |
| `evaluate_mujoco.py` | Lặp controller trên `--seeds` → `outputs/controller_evaluation/`: episode JSON, CSV, summary. | Hiện chỉ hỗ trợ `--mode controller`; chưa nạp checkpoint ACT. |

Ví dụ kiểm nhanh và tạo báo cáo vào thư mục mới:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_mujoco.py --output docs/audit/my_camera_check.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/run_pick_place_demo.py --viewer --video --output outputs/my_demo
```

Các script có `argparse` cho phép xem `--help`; script helper/ví dụ có thể không có CLI riêng. Đã có `train.py` và `validate.py` riêng của dự án cho CFG5 pretraining; chưa có `test.py` hoặc learned ACT rollout. Đừng nhầm `third_party/rh20t_act/train.py` với trainer mới. Quy trình dữ liệu đầy đủ ở [hướng dẫn pipeline](../docs/PIPELINE_GUIDE_VI.md); [hướng dẫn pretraining](../docs/PRETRAINING_VI.md) ghi lệnh server.

| Script mới | Vai trò |
|---|---|
| `train.py` | Đọc ACTDataset train/val, chạy optimizer, validation toàn tập, lưu `latest.pt`/`best.pt` và resume tại ranh giới epoch. `--smoke` chỉ chạy một batch mỗi split. |
| `validate.py` | Nạp checkpoint đúng contract và tính offline validation, ghi số sample/batch/episode/target thật. Không cập nhật model. |

## Bổ sung lượt4a — ACT và dữ liệu người dùng tự chuẩn bị

| File | Input → output; logic |
|---|---|
| `download_act_baseline.py` | Commit khóa → third_party/rh20t_act và manifest; kiểm Git blob/SHA-256, không tải weights. |
| `check_act.py` | Model config+train smoke → docs/audit/act_integration_check.json; full inference và backward cấu hình nhỏ, không optimizer.step. |
| `review_cfg5.py` | Manifest/split → outputs/cfg5_review/review.html, summary.json, selection.json; bảng accepted/rejected/link video, không full decode. Không ghi đè selection. |
| `select_cfg5.py` | Manifest+JSON allowlist → manifest dẫn xuất mới có provenance/hash; chặn ID lạ/trùng, không sửa raw hoặc split cũ. |
| `inspect_processed.py` | --episode-dir và --index → index.html, eight_cameras.png, sample.json; memmap arrays, xuất một mốc với target/mask ở đơn vị vật lý. |

Lệnh cụ thể và cách chọn dữ liệu: [CFG5_HANDS_ON_VI](../docs/CFG5_HANDS_ON_VI.md). Cài model: [ACT_SETUP_VI](../docs/ACT_SETUP_VI.md).

## Tải thư mục Google Drive (2026-10-08)

`download_drive.py` là entrypoint độc lập trên Linux, dùng rclone remote đã xác thực.
Các lệnh `start/status/stop/verify` tải 4 file song song, chạy nền, resume theo file
hoàn chỉnh, retry và đối chiếu nguồn/đích trước khi báo hoàn tất. Không đổi trainer,
không tự tải dataset khi cài gói. Xem [hướng dẫn](../docs/DRIVE_DOWNLOAD_VI.md).
