# Khảo sát workspace — lượt 1

Ngày: 2026-09-27. Project root: `D:\Download\mophong_RH20T`.

Đây là snapshot lượt1. Trạng thái mới nhất và môi trường `.venv-data` nằm trong [báo cáo lượt2](STAGE2_REPORT_VI.md).

## 1. Trạng thái trước khi thay đổi

- Root chỉ có `RH20T_cfg5/`; chưa có code, config, model, checkpoint hoặc môi trường dự án tìm thấy qua khảo sát file.
- `git status --short` trả về “not a git repository”. Chưa khởi tạo Git hoặc tạo commit.
- Tại thời điểm khảo sát ban đầu chưa có tài liệu quy trình nội bộ tại root.
- Hai PDF ban đầu nằm ngoài root tại `D:\Download`; đã truy cập theo đường dẫn người dùng bổ sung.
- Python mặc định: 3.12.7. Kiểm tra module bằng `find_spec` không tìm thấy NumPy, PyTorch, MuJoCo, pypdf hoặc PyMuPDF trong interpreter này.
- Runtime có sẵn của Codex có NumPy và pypdf; dùng để đọc PDF và các mẫu NPY. Không cài thêm dependency vào Python của người dùng.

## 2. Dataset hiện có

Đường dẫn giữ nguyên: `RH20T_cfg5/`. Quét metadata filesystem, không giải mã toàn bộ video hoặc đọc toàn bộ dataset vào RAM.

| Thông tin | Kết quả |
|---|---:|
| Thư mục đúng mẫu `task_*_user_*_scene_*_cfg_0005` | 1.321 robot |
| Thư mục cùng mẫu thêm `_human` | 1.191 human |
| Thư mục biến thể `_human_2` | 3, để riêng chờ xử lý naming |
| Tổng thư mục bắt đầu `task_` | 2.515 |
| Task ID khác nhau ở nhóm đúng mẫu | 104 |
| User ID ở nhóm đúng mẫu | 0007, 0010, 0014, 0020, 0062 |
| Tổng file kể cả calibration | 56.202 |
| Tổng dung lượng từ file metadata | 92.876.723.799 bytes, khoảng 86,50 GiB |
| Video `color.mp4` | 19.340 |
| File `depth.mp4` | 0 |
| Calibration | Có thư mục `calib/` |

Ba thư mục biến thể:

```text
task_0001_user_0014_scene_0001_cfg_0005_human_2
task_0050_user_0007_scene_0005_cfg_0005_human_2
task_0062_user_0007_scene_0001_cfg_0005_human_2
```

Số đếm sơ bộ ban đầu phân loại bằng đuôi `_human` đã gộp nhầm 3 thư mục này vào robot; bảng trên là kết quả sau khi kiểm tra mẫu tên nghiêm ngặt. Không tự đổi tên hoặc xóa chúng.

Có 4 thư mục robot không chứa `transformed/joint.npy`:

```text
task_0039_user_0007_scene_0005_cfg_0005
task_0043_user_0007_scene_0009_cfg_0005
task_0060_user_0007_scene_0008_cfg_0005
task_0106_user_0007_scene_0005_cfg_0005
```

Đây là kiểm tra sự tồn tại file; chưa kết luận nguyên nhân là download thiếu hay upstream không cung cấp. Không yêu cầu tải lại dataset dựa trên dấu hiệu này.

## 3. Mẫu thực tế đã đọc

Robot: `task_0001_user_0007_scene_0001_cfg_0005`.

```json
{"finish_time": 1640002341257, "rating": 9, "calib_quality": 2,
 "calib": 1640002326600, "action": 1639120789274}
```

`action` trong metadata ở đây là một giá trị scalar giống timestamp/ID; không diễn giải nó là vector điều khiển.

Human: `task_0001_user_0010_scene_0001_cfg_0005_human`. Mẫu này có `start_time`, `end_time`, `calib`, `calib_quality`; không có thư mục `transformed`. Hai mẫu robot/human này dùng để khảo sát định dạng, **không phải một cặp tương ứng**, vì khác user ID.

| File/nhóm | Dạng thực tế trong mẫu | Giới hạn kết luận |
|---|---|---|
| `cam_*/color.mp4` | Có video ở nhiều camera | Chưa decode; chưa xác minh W/H/FPS và frame count video |
| `timestamps.npy` | Scalar object array chứa dict `color`/`depth`, mỗi mục là list timestamp | Có timestamp depth nhưng không có depth.mp4 trên máy |
| Camera `036422060215` | 231 timestamp color và 231 depth; median delta=68 ms | Dữ liệu mẫu không phải lưới đều 10 Hz; cần resampling/đồng bộ ở lượt 2 |
| `joint.npy` | Dict camera → dict timestamp → float64 array **14 phần tử** | API CFG5 lấy `[0:7]` và map sang panda_joint1…7; 7 phần tử sau chưa gán nghĩa trong dự án |
| `tcp.npy`, `tcp_base.npy` | Dict camera → list record gồm `timestamp`, `tcp` (7), `robot_ft` (6) | Khác frame; chưa xác minh mọi transform với MuJoCo |
| `gripper.npy` | Dict camera → dict timestamp → `gripper_command` và `gripper_info`, mỗi giá trị 3 phần tử | Command và measured info khác nhau; chưa chốt scale/meaning cho adapter |
| `high_freq_data.npy` | Dictionary rỗng ở mẫu robot này | Không giả định mọi episode có stream 100 Hz |
| `robot_command/tcpcommand_timestamp.npy` | Numeric float64 array `(193580,7)`, có cột cuối giống timestamp | Chưa xác minh units/frame, thời gian hiệu lực, 6 cột đầu hoặc command latency |

Các NPY nhiều modality là object arrays, không phải tensor float có thể mmap trực tiếp. Chế độ khảo sát mặc định chỉ đọc header; chế độ payload được bật có chủ đích cho các file dataset cục bộ đã biết, giới hạn 5 MB/file. File robot command lớn hơn giới hạn chỉ có header trong báo cáo script; shape và hai hàng đầu đã được đọc riêng ở khảo sát thủ công.

**Điều chưa xác minh:** completeness so với archive chính thức; video decode; đồng bộ mọi camera; đơn vị và quaternion ordering trên toàn pipeline; schema 7 phần tử sau của joint; task subset dùng train; những episode lỗi/NaN; calibration quality và mapping TCP-to-tool trong sim. Không dùng suy đoán để lấp các mục này.

## 4. File đã tạo trong lượt này

- `README.md`, `.gitignore`, `requirements.txt` cho trạng thái khảo sát.
- `references/` gồm hai PDF bản sao, manifest hash, nguồn, ghi chú, bản yêu cầu và snapshot bốn repository kèm license.
- `docs/PAPER_ANALYSIS.md`, `docs/ACT_IMPLEMENTATION_REVIEW.md`, `docs/DECISIONS.md`, báo cáo này.
- `scripts/inspect_data.py`: công cụ chỉ đọc raw data, ghi báo cáo ngoài raw.
- `scripts/probe_validation_sampling.py`: probe source sampling cô lập với dữ liệu giả.
- `docs/audit/`: inventory, payload samples và kết quả probe nhỏ.

Không có preprocessing, model ACT, environment MuJoCo hoặc training run được triển khai ở lượt này. `references/upstream/` không phải working copy dependency; không tuyên bố model upstream đã chạy.

## 5. Những gì đã kiểm chứng và lệnh chạy lại

Chạy mọi lệnh từ `D:\Download\mophong_RH20T`.

### Kiểm kê bằng Python mặc định

```powershell
python scripts/inspect_data.py --scan-files
```

Input: `RH20T_cfg5/`. Output: `docs/audit/project_inventory.json`. Đã chạy thành công. Đọc header và đếm filesystem; không đọc nội dung video. Số liệu dự kiến như bảng trên nếu dataset không đổi.

### Đọc payload nhỏ và probe upstream bằng runtime đã có

```powershell
$auditPython = 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $auditPython scripts/inspect_data.py --trusted-npy-samples --output docs/audit/payload_samples.json
& $auditPython scripts/probe_validation_sampling.py
```

Runtime này do Codex cung cấp, có thể thay đổi khi app cập nhật; về sau có thể thay bằng Python môi trường dự án có NumPy. Input của probe là snapshot source + timestamp giả. Output: `docs/audit/upstream_sampling_probe.json`. Đã chạy thành công; `padding_bug_reproduced=true` là **xác nhận tìm thấy lỗi upstream**, không phải phép thử cho rằng mask upstream đúng.

Kiểm tra source hash và PDF hash đã được thực hiện: 51 file ở 4 repository khớp manifest, hai PDF copy trùng nội dung bản gốc. Đã kiểm tra cú pháp hai script, 12 liên kết tài liệu tương đối và thử đặt output vào raw: script trả exit code 2, không tạo file. Báo cáo kiểm tra nằm ở `docs/audit/verification.json`.

Script inventory đã chạy được trong interpreter không có NumPy ở chế độ mặc định; chế độ payload dùng NumPy riêng. Không test GPU, ACT forward/backward, MuJoCo hoặc robot motion ở giai đoạn này.

Lỗi thường gặp: nếu Python mặc định chưa có NumPy, không dùng cờ payload/probe bằng interpreter đó. Nếu thay đổi snapshot source, probe từ chối khi hash không khớp. Nếu đặt output trong raw, inventory từ chối trước khi ghi.

## 6. Bước tiếp theo

Chốt lựa chọn upstream và action representation trong [DECISIONS](DECISIONS.md), sau đó triển khai khảo sát sâu, manifest sạch, episode split, preprocessing và dataset adapter với test dữ liệu giả. Những bước này cần giữ các khác biệt với baseline chính thức có thể truy vết.
