# Báo cáo lượt 2 — dữ liệu pretraining và rig camera

Ngày2026-09-27. Đã triển khai theo mục tiêu cập nhật: CFG5 để pretraining ACT, task chính là đặt vật vào khuôn nhỏ trong MuJoCo với8 camera. Không chạy training dài, không sửa dữ liệu gốc và không sửa upstream.

## Kết quả đã kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Khảo sát sâu | 99 robot episode thuộc task4/8/13/14/17; 57 đạt bộ lọc strict8 RGB view, 42 bị loại |
| Split | 47 train, 5 val, 5 test; group `(task,user,calib)`, seed42 |
| Timeline | 5.473 bước sau lọc, 684 segment; 4.789 sample có future target theo timeline |
| Unit tests | 8/8 đạt: causal indexing, gap, chunk boundary, video mismatch, quaternion, future target/RGB/mask, train-only stats/split, fixed/wrist movement |
| Smoke thật | 1 train episode và1 val episode, mỗi episode8 bước; khoảng88.5 MB ảnh |
| Dataset check | Train6 sample/val7 sample; đọc đầu/giữa/cuối; images `(8,3,360,640)`, actions `(20,8)`, mask bool |
| Statistics | Chỉ1 train smoke episode; `smoke_only:true`; val dùng đúng statistics này |
| Camera | 8 roundtrip calibration; 6 fixed/2 wrist movement; 60 pose crosschecks (12 camera/calibration ×5) chỉ trên train |
| Dependencies | Windows CPython3.12.14, NumPy2.5.3, OpenCV-headless4.14.0.94; pip check không lỗi |

57 episode là subset pretraining ban đầu, không phải toàn bộ CFG5. 39 episode thiếu camera bắt buộc, 2 rating dưới ngưỡng, 1 thiếu khoảng giao đủ dài. Median segment6 bước; chỉ275 sample có đủ20 future targets. Filter8-view nghiêm ngặt đánh đổi số lượng dữ liệu lấy input thống nhất với rig đích.

## Code và artifacts

- `src/data/`: timestamp/schema checks, streaming RGB preprocessing, group split, train-only normalization, NumPy ACTDataset.
- `scripts/`: prepare_manifest, create_splits, preprocess_data, fit_normalization, check_dataset, export_camera_rig, audit_camera_rig.
- `configs/data/cfg5_base8.json`: representation, camera order, tasks, filter, rate, chunk và split parameters.
- `configs/mujoco/cfg5_camera_rig.json`: calibrated transforms/intrinsics; `precision_pick_place.json`: thiết kế hình học và metric ban đầu.
- `data/splits/`: manifest, split và smoke statistics. `data/processed/smoke_cfg5/` chứa smoke arrays, được ignore khỏi Git.
- `docs/audit/train_dataset_smoke.json`, `val_dataset_smoke.json`, `camera_rig_check.json`: bằng chứng chạy thật.
- `docs/DATA_GUIDE_VI.md`, `CAMERA_SETUP_VI.md`, `PIPELINE_GUIDE_VI.md`, `EXPERIMENT_PROTOCOL.md`, `DECISIONS.md`: cách chạy, toán học, protocol và giới hạn.

Kho tài liệu tiếp tục dùng `references/papers/`, `references/notes/`; nguồn/commit/license có trong `references/SOURCES.md` và manifest upstream. Không tạo project lồng hoặc di chuyển raw.

## Chạy lại kiểm thử

```powershell
Set-Location 'D:\Download\mophong_RH20T'
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/audit_camera_rig.py
```

Kỳ vọng8 tests OK, camera checks8 và fixed-camera checks12. Lệnh smoke/full preprocessing chi tiết nằm trong [PIPELINE_GUIDE_VI.md](PIPELINE_GUIDE_VI.md); output có sẵn sẽ bị từ chối ghi đè.

## Giới hạn và bước tiếp theo

Chưa giải mã toàn57 episode (ước tính30.26 GB ảnh), chưa có model forward/training/test loss. Chưa tải Panda asset, cài MuJoCo, dựng PandaEnv, render8 view, controller hay evaluator vật lý. Camera rig là calibration toán học, chưa chứng minh mọi camera nhìn rõ khuôn/không che khuất trong model mô phỏng. TCP mount, lens projection và cavity contact còn phải kiểm lượt3.

Lượt3 sẽ dựng mô phỏng chạy thật theo rig này, kiểm controller và đặt vào khuôn; lượt4 tích hợp ACT/pretraining/fine-tuning. Không cần người dùng chốt lại representation/camera vì đã giao quyền lựa chọn. Việc chia lượt tuân theo mục10 yêu cầu gốc, không phải yêu cầu cấp quyền thêm.
