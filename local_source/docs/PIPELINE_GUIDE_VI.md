# Pipeline thực hành

Mọi lệnh chạy trong PowerShell tại `D:\Download\mophong_RH20T`. Mục1–6 ghi lại pipeline dữ liệu lượt2; môi trường hiện đã thêm MuJoCo ở lượt3. Xem mục7 và [hướng dẫn mô phỏng](MUJOCO_PANDA_GUIDE_VI.md) cho trạng thái mới. Kiểm thử logic đặt vật mới nhất được ghi trong [PLACEMENT_LOGIC_REVIEW_VI.md](PLACEMENT_LOGIC_REVIEW_VI.md).

## 1. Môi trường hiện có

```powershell
Set-Location 'D:\Download\mophong_RH20T'
& '.\.venv-data\Scripts\python.exe' --version
& '.\.venv-data\Scripts\python.exe' -m pip check
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
```

Mong đợi Python3.12.14, không lỗi dependency và toàn bộ tests OK. Môi trường hiện có NumPy, OpenCV-headless và MuJoCo theo requirements-sim-lock.txt; đã bổ sung Torch CPU cho ACT integration; lock đầy đủ ở requirements-act-lock.txt. `-X utf8` tránh lỗi in tiếng Việt trên terminal Windows. Không cần activate venv.

`python` trên PATH của máy hiện là MSYS2 Python3.12.7. Không dùng lẫn nó với `.venv-data`. Nếu cần tạo lại môi trường ở máy hiện tại, dùng standard Windows CPython đang có:

```powershell
& 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv-data
& '.\.venv-data\Scripts\python.exe' -m pip install -r requirements-data-lock.txt
```

Chỉ tạo lại nếu môi trường chưa tồn tại. Trên máy khác, thay đường dẫn Python đầu bằng standard CPython3.12 tương ứng. Không tải model/backbone ở bước này. `.venv/` cũ do MSYS2 tạo không dùng trong pipeline.

## 2. Tài liệu và inventory

Đọc `docs/DECISIONS.md`, `docs/PAPER_ANALYSIS.md`, `docs/ACT_IMPLEMENTATION_REVIEW.md`; hai file sau mô tả khảo sát lượt1, lựa chọn mới nhất nằm ở DECISIONS. Tài liệu tham khảo mới đặt vào `references/papers/`, ghi chú vào `references/notes/`.

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/inspect_data.py --scan-files
```

Input raw `RH20T_cfg5/`; output `docs/audit/project_inventory.json`. Kỳ vọng inventory robot1321/human1191 và104 task như lượt1, không đồng nghĩa mọi episode sử dụng được. Nếu không tìm thấy raw, kiểm working directory và config; không tải dataset lại.

## 3. Manifest và split

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/prepare_manifest.py
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/create_splits.py
```

Input `configs/data/cfg5_base8.json` và raw; output `data/splits/cfg5_manifest.json`, `cfg5_splits.json`. Kỳ vọng accepted57/rejected42 và split47/5/5. Chi tiết lý do loại ở manifest. Split theo group trước chunking, không tách frame ngẫu nhiên.

Nếu đổi bất cứ trường config, hash cũng đổi; tạo lại artifacts vào vị trí thích hợp và không dùng processed/statistics cũ. Sau khi bắt đầu nghiên cứu chính thức, khóa config/split thay vì liên tục ghi đè chúng. Manifest kiểm frame đầu và metadata; full decode có thể phát hiện lỗi tiếp.

## 4. Smoke preprocessing nhỏ

Output đã có của lượt2 là `data/processed/smoke_cfg5/`, không ghi đè. Để chạy lại, chọn thư mục mới, ví dụ `smoke_cfg5_recheck` nếu chưa tồn tại:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --split train --limit-episodes 1 --max-steps 8 --output-root data/processed/smoke_cfg5_recheck
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --split val --limit-episodes 1 --max-steps 8 --output-root data/processed/smoke_cfg5_recheck
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/fit_normalization.py --processed-root data/processed/smoke_cfg5_recheck --output data/splits/smoke_normalization_recheck.json --allow-partial-smoke
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --processed-root data/processed/smoke_cfg5_recheck --stats data/splits/smoke_normalization_recheck.json --split train --allow-partial-smoke --output docs/audit/train_dataset_recheck.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --processed-root data/processed/smoke_cfg5_recheck --stats data/splits/smoke_normalization_recheck.json --split val --allow-partial-smoke --output docs/audit/val_dataset_recheck.json
```

Kỳ vọng 1 episode mỗi split ×8 bước; statistics chỉ từ train; train6 samples (hai segment), val7 samples; ảnh `(8,3,360,640)`, actions `(20,8)`. Script check đọc đầu/giữa/cuối dataset, không giả vờ duyệt mọi sample. Báo cáo có `model_executed:false`. Đây là kiểm preprocessing/adapter, không phải validation ACT.

Nếu báo output/partial đã có, chọn tên thư mục mới. Nếu statistics báo train chưa đủ, smoke cần cờ `--allow-partial-smoke`; không dùng cờ đó để bỏ qua thiếu dữ liệu trong thí nghiệm thật. Decode failure giữ ERROR.json trong partial; raw vẫn nguyên vẹn.

## 5. Preprocessing chính thức — người dùng chạy khi sẵn sàng

Toàn subset dự kiến khoảng30.26 GB ảnh uncompressed, cộng metadata. Lượt2 chưa chạy phần này. Chọn output root còn đủ dung lượng, sau đó:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --split train
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/preprocess_data.py --split val
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/fit_normalization.py
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --split train --output docs/audit/train_dataset_full.json
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_dataset.py --split val --output docs/audit/val_dataset_full.json
```

Output `data/processed/cfg5/` và `data/splits/normalization.json`, `smoke_only:false`. Nếu mọi episode decode được, dataset train3883/val477 samples **trong subset cũ**. Không có cờ smoke/limit. **Cập nhật CLI:** thêm `--resume` để kiểm và bỏ qua episode hoàn chỉnh; `.partial` vẫn phải xem `ERROR.json`, không xóa raw. Hướng dẫn full CFG5 mới ở [CFG5_VSCODE_SMOKE_VI](CFG5_VSCODE_SMOKE_VI.md). Full decode chưa được kiểm thử toàn bộ nên có thể phát hiện hỏng video ở giữa.

Test split đã khóa nhưng chưa đánh giá model. Khi checkpoint/config đã chốt ở lượt4 mới dùng `preprocess_data.py --split test` và test entrypoint của lượt4; không dùng test loss để điều chỉnh config.

## 6. Rig camera

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/export_camera_rig.py
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/audit_camera_rig.py
```

Input calibration thật + upstream config5. Output `configs/mujoco/cfg5_camera_rig.json`, `docs/audit/camera_rig_check.json`. Kỳ vọng 8 cameras và12 fixed-camera TCP crosschecks. Xem `docs/CAMERA_SETUP_VI.md`. Test này không khởi tạo MuJoCo.

## 7. MuJoCo đã triển khai; ACT thuộc lượt4

Lượt3 đã cung cấp model Panda/licensing, scene với3 chốt và3 lỗ Ø7mm,8 policy camera, PandaEnv, controller, viewer/headless, evaluator hình học/contact và smoke demo. Chạy `scripts/check_mujoco.py` rồi `scripts/run_pick_place_demo.py --video` bằng Python .venv-data. [MUJOCO_PANDA_GUIDE_VI.md](MUJOCO_PANDA_GUIDE_VI.md) có đầy đủ lệnh cài/chạy/API và kết quả. Config chính là `configs/mujoco/panda_three_pegs.json`; precision_pick_place.json hiện chỉ là alias tới config chính, không còn kích thước khối hộp cũ. Không gửi action8D trực tiếp vào actuator.

ACT model đã tích hợp và smoke kiểm; xem [ACT_SETUP_VI.md](ACT_SETUP_VI.md). Theo thứ tự mới, [CFG5_HANDS_ON_VI.md](CFG5_HANDS_ON_VI.md) hướng dẫn người dùng chọn/chuyển/xem dữ liệu trước pretraining. Trainer/validate/test/checkpoint/resume và demonstrations/ACT rollout vẫn là việc tiếp theo. Trình tự: pretrain CFG5 → fine-tune demonstrations khuôn → chọn checkpoint theo val → khóa config/checkpoint → test offline và rollout cuối. Dùng `docs/EXPERIMENT_PROTOCOL.md` để so sánh với target-only. Không coi demo controller thành kết quả ACT, không có success rate ACT ở giai đoạn này.
