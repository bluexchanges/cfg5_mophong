# Quy tắc làm việc trong dự án RH20T

Trạng thái 2026-09-30: trainer/validation/checkpoint/resume CFG5 đã thêm trong `src/training/` và `scripts/train.py`, `scripts/validate.py`; test fixture nhỏ đã chạy, full ACT GPU chưa chạy. `docs/PRETRAINING_VI.md` là hướng dẫn mới. Bản processed trên máy A đã dọn; server sẽ preprocess lại. `local_source` là workspace riêng có raw/rule, `github_source` chỉ nhận code/config/tài liệu công khai và metadata nhỏ, không có AGENTS.md, raw, cache, môi trường hoặc output lớn.

- Đọc `references/requests/PROJECT_INSTRUCTIONS_VI.md` và `docs/DECISIONS.md` trước khi triển khai giai đoạn tiếp theo. Hướng dẫn trực tiếp mới nhất của người dùng có ưu tiên cao hơn.
- Viết tài liệu và báo cáo bằng tiếng Việt, giữ thuật ngữ kỹ thuật cần thiết; giải thích cho người mới học robot learning.
- Khảo sát cây thư mục, Git status (nếu có), dữ liệu, tài liệu và implementation hiện hữu trước khi sửa.
- Giữ nguyên nội dung payload CFG5, không giải nén thêm hoặc load toàn bộ dataset vào RAM. Ngoại lệ người dùng đã yêu cầu ngày2026-09-28: move toàn bộ episode vào `training_data/` và `missing_data/` theo task, không copy; đã verify2515 move. CSV chỉ mục human–robot nhỏ được phép lưu trong CFG5 để giữ quan hệ sau move. Không move/sửa/xóa payload tiếp nếu không có yêu cầu mới.
- Paper gốc nằm ở `references/papers/`; tra nguồn và phiên bản trong các manifest. Phân biệt kết luận từ paper, hành vi source code và quyết định của dự án.
- `references/upstream/` là snapshot phục vụ khảo sát, không phải dependency đã cài. Giữ nguyên file upstream; không sửa để chạy thí nghiệm. Dùng adapter/wrapper ngoài upstream, ghi patch nếu thực sự cần.
- Không tự chạy training dài, rollout dài hoặc tải checkpoint/model asset lớn. Thông báo trước khi cài dependency lớn hoặc tải asset.
- Người dùng đã giao quyền tự chọn representation và cấu hình phù hợp CFG5. Ghi căn cứ trong DECISIONS.md, triển khai không hỏi lại các lựa chọn kỹ thuật thuộc phạm vi này. Mục tiêu: CFG5 để pretraining, ACT fine-tuning cho pick-place vào khuôn nhỏ; mô phỏng dùng 8 camera dựa trên calibration thật.
- Không coi measured state là command nếu chưa xác minh. Không gọi 14 phần tử của `joint.npy` là 14 khớp; dùng schema đã kiểm tra.
- Split theo episode/group trước khi tạo action chunks; mọi camera và human counterpart của cùng episode thuộc cùng split. Chỉ tính thống kê học từ train. Test chỉ dùng sau khi khóa checkpoint và config.
- Báo cáo rõ validation sample/batch/episode thực tế và padding denominator. Offline loss, controller demo và ACT simulation rollout là các kết quả khác nhau.
- Chỉ tuyên bố chức năng đã chạy khi có kiểm thử. Ghi seed, config, upstream commit, dữ liệu đầu vào, output và giới hạn kiểm chứng.
- Không commit dataset, checkpoint, video, cache hoặc môi trường ảo. Giữ code/config/tài liệu cần tái lập trong Git.

Lựa chọn chính: lõi RH20T ACT baseline (đã tích hợp model, inference/backward smoke; chưa training), TCP base xyz+wxyz+gripper width mét,8D; RGB8 camera. Future pose CFG5 là measured-state proxy. Lượt3 đã có PandaEnv/MuJoCo và controller đặt3 chốt Ø6mm vào3 lỗ Ø7mm theo màu; config chính `configs/mujoco/panda_three_pegs.json`. Không sửa upstream Panda XML; sửa builder/config rồi regenerate scene. TCP ảo mô phỏng chưa phải factory EE transform CFG5 đã đo. Theo yêu cầu2026-09-28: tích hợp ACT trước → người dùng chọn/preprocess CFG5 → pretraining; demonstrations task đích làm sau. Đã có third_party/rh20t_act và src/models/rh20t_act.py; chưa trainer/ACT rollout. Dùng `.venv-data/Scripts/python.exe -X utf8`; Python trên PATH là MSYS2. Dependency dữ liệu/mô phỏng ở requirements-sim-lock.txt; ACT CPU đã kiểm ở requirements-act-lock.txt.

Logic đặt vật schema2: chốt35mm, lỗ sâu12mm, độ nhô mục tiêu23±0.5mm; kiểm riêng đáy, lực đỡ, phần chốt trong lỗ, nghiêng và nhả robot. `precision_pick_place.json` chỉ là alias. Xem `configs/mujoco/README.md` và `docs/PLACEMENT_LOGIC_REVIEW_VI.md` trước khi sửa evaluator.

Đọc docs/ACT_SETUP_VI.md và docs/CFG5_HANDS_ON_VI.md cho lượt4a. Giữ cả source third_party/rh20t_act nguyên vẹn, adapter kiểm hash. Không chạy full preprocessing hoặc pretraining thay người dùng trong lượt chỉ hướng dẫn.

Sau phân loại toàn CFG5, đọc `docs/CFG5_FULL_FILTER_VI.md`. Config mới `configs/data/cfg5_all8.json`, manifest `data/splits/cfg5_all_manifest.json`, chỉ mục human–robot `RH20T_cfg5/human_robot_index.csv` và `data/index/cfg5_human_robot_pairs.json`. Quan hệ theo tên chỉ là ứng viên, không là action label đồng bộ cho human frame. Các số57/47/5/5 thuộc subset lịch sử, không phải toàn CFG5.

Split full đã khóa tại `data/splits/cfg5_all_splits.json`: ngẫu nhiên seed42 theo group `(task,user,calib)`, đúng531/66/66 episode. Không chọn lại seed sau khi xem test. Người dùng tự chạy preprocessing; chỉ split trong lượt này, không tự chuyển đổi toàn bộ. Lệnh, dung lượng và cơ sở đồng bộ ở `docs/CFG5_FULL_SPLIT_AND_RUN_VI.md`.

Preprocess CLI đã bổ sung `--resume` để kiểm và bỏ qua episode hoàn chỉnh; `.partial` vẫn dừng và cần kiểm ERROR.json. Hướng dẫn người dùng thao tác VS Code/smoke ở `docs/CFG5_VSCODE_SMOKE_VI.md`. Không tự chạy full preprocessing thay người dùng.

Người dùng đã tự chạy xong full preprocessing531 train+66 val tại `data/processed/cfg5_all/` (flat layout; membership theo split JSON), test66 chưa xử lý. Audit `docs/audit/cfg5_all_processed_postrun.json` kiểm đủ597 episode,4.776 array headers và metadata; không lỗi/partial. Train-only normalization `data/splits/cfg5_all_normalization.json` đã fit trên531 episode, smoke_only=false. `check_dataset` train/val đã kiểm sample đầu/giữa/cuối; chưa train ACT. Không chuyển hoặc copy400GB chỉ để tách thư mục vật lý; `split_membership.csv` nhỏ trong processed root cho người dùng tra Explorer.
