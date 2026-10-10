# Lọc toàn CFG5 và giữ quan hệ human–robot

Ngày 2026-09-28, người dùng yêu cầu dùng **toàn bộ CFG5** thay vì chỉ 99 robot episode thuộc 5 task. Chúng tôi đã kiểm cả 2.515 thư mục episode trên 104 task rồi **chuyển thư mục trên cùng ổ D:**, không sao chép payload. Mã chạy, quyết định và biên bản lần lượt ở `scripts/organize_cfg5.py`, `configs/data/cfg5_all8.json`, `audit/cfg5_all_organization_plan.json` và `audit/cfg5_all_organization_verified.json`.

## Cây dữ liệu hiện tại

```text
RH20T_cfg5/
  human_robot_index.csv             # Toàn bộ ứng viên liên hệ human–robot
  training_data/task_XXXX/<robot_episode>/
  missing_data/task_XXXX/
    human_robot_index.csv           # Chỉ mục riêng của task, mở bằng Excel/Notepad
    human_for_future_one_shot/<human_episode>/
    robot_incomplete/<robot_episode>/
  calib/ ...                        # Calibration gốc giữ nguyên vị trí
```

`missing_data` nghĩa là **không đủ đầu vào/nhãn cho pipeline ACT 8D hiện tại**, không có nghĩa video human hỏng hay vô giá trị. Chúng được giữ nguyên để nghiên cứu human-conditioned/one-shot sau này. CSV dùng đường dẫn tương đối tính từ `RH20T_cfg5/`, ghi rõ robot episode tương ứng đang nằm ở `training_data`, `robot_incomplete`, hay vắng trên máy; có calibration ID của mỗi bên và cột `same_calibration`. JSON có cùng bảng tại `data/index/cfg5_human_robot_pairs.json`. Không cần mò từng tên thư mục.

**Quan hệ trong CSV chỉ là ứng viên theo cùng mã task/user/scene/cfg.** Chưa có bằng chứng hai episode là cùng một lần thao tác, cùng calibration hoặc đồng bộ theo frame. Ví dụ cặp đầu task 0001 có `start_time` human vào tháng 1/2022 nhưng robot `finish_time` vào tháng 12/2021 và calibration ID khác nhau. Vì thế không dùng pose robot tại frame t làm nhãn trực tiếp cho frame human t. Nếu sau này huấn luyện one-shot, cần định nghĩa pairing theo mục tiêu/task và phương pháp học riêng, rồi xác minh quan hệ bằng nội dung và metadata.

## Tiêu chí lọc ACT hiện tại

Robot episode được nhận khi rating ≥2, có `transformed/tcp_base.npy` và `transformed/gripper.npy`, đủ 8 camera RGB và timestamp trong thứ tự config, video mở được và số frame khớp timestamp. Timeline phải tạo được lưới 10 Hz với lấy mẫu từ quá khứ, tuổi ảnh/state tối đa 150 ms, pose/width hợp lệ và segment liên tục ít nhất 3 bước. Action arm là **future measured-state proxy**, không khẳng định là lệnh điều khiển thật. Không tự chèn camera thiếu hoặc nối qua khoảng mất đồng bộ.

| Phân loại | Số episode | Diễn giải |
|---|---:|---|
| `training_data` | 663 robot trên 92 task | Đạt bộ lọc ACT 8 camera/8D. |
| `missing_data/.../robot_incomplete` | 658 robot | 483 thiếu file bắt buộc; 116 rating dưới 2; 59 không qua kiểm video/timeline/state. Lý do chi tiết theo episode trong plan. |
| `missing_data/.../human_for_future_one_shot` | 1.194 human | Có video human; không có target robot 8D trực tiếp cho ACT. Không đánh đồng với robot incomplete. |

Chỉ mục human: 606 ứng viên tên trùng robot đã đạt ACT, 587 trùng robot chưa đạt, 1 không thấy robot trên máy. Trong 1.193 ứng viên có robot, **260 cùng calibration và 933 khác calibration**; cùng calibration cũng chưa chứng minh đồng bộ. Ba thư mục hậu tố `_human_2` được giữ và ghi thành ba dòng riêng. Tổng dữ liệu episode trước/sau move là **56.066 file, 92.704.220.008 byte**; thêm calibration giữ nguyên ở root là **56.202 file, 92.876.723.799 byte**. Sau move, mỗi folder episode còn cùng directory identity, số file và số byte như plan; không còn episode ở root. CSV/JSON index mới là metadata nhỏ, không phải bản copy video.

Timeline của 663 robot episode được nhận có 80.015 bước/8.721 segment. Suy ra khoảng 71.294 observation có ít nhất một future target và 5.301 observation có đủ 20 target nếu **toàn bộ** được preprocess thành công. Đây là ước tính ban đầu từ kiểm timeline, không phải số sample NPY đã lưu hay số sample model đã học. Chỉ riêng RGB uint8 chưa nén ở 640×360×8 view được ước tính khoảng **442,45 GB** cho cả 663 episode. Sau bước này, người dùng đã preprocess 531 train và 66 validation episode (khoảng 400,33 GB); 66 test episode chưa xử lý. Xem [báo cáo kiểm sau xử lý](audit/cfg5_all_processed_postrun.json) để biết số thực tế. Chưa pretrain ACT.

## Paper dùng human như thế nào?

RH20T paper, mục III-A và hình 4, mô tả cấu trúc thứ bậc human demonstration/robot trajectory và động cơ one-shot. Tuy nhiên, thí nghiệm ACT ở mục IV-A dùng **robot trajectories**: 335 demo robot cùng task, 195 demo robot từ ba task liên quan, rồi fine-tune với 75/40/10 demo robot của môi trường mới. Paper không báo cáo việc đưa video human thành action targets trực tiếp cho ACT. Source baseline cũng đọc image cùng TCP/gripper robot. Đây là giới hạn của thí nghiệm ACT được báo cáo, không phải lý do xóa human data.

## Dùng file nào tiếp theo?

1. Mở `RH20T_cfg5/human_robot_index.csv` để tìm human, robot và trạng thái. Lọc cột `task` hoặc mở CSV trong `missing_data/task_XXXX/`.
2. Đọc `docs/audit/cfg5_all_organization_plan.json` để xem lý do lọc từng robot episode và quality của episode được nhận.
3. `data/splits/cfg5_all_manifest.json` là manifest của toàn bộ 104 task, với 663 accepted và 658 rejected robot. Split full đã tạo tại `data/splits/cfg5_all_splits.json`, đúng531/66/66 episode; hướng dẫn ở [CFG5_FULL_SPLIT_AND_RUN_VI](CFG5_FULL_SPLIT_AND_RUN_VI.md). `cfg5_manifest.json`/`cfg5_splits.json` cũ chỉ là lịch sử 5 task/57 episode.
4. Khi bắt đầu pretraining, chia theo episode/group trước khi tạo action chunks, tính statistics từ train, khóa test. Nếu dùng human cùng robot trong một thí nghiệm mới, gán split theo quan hệ trong index để tránh leakage. Chưa dùng human để fit ACT 8D ở giai đoạn này.

Lệnh chỉ đọc để xác nhận lại move: `& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/organize_cfg5.py --verify`. Script chỉ mục: `& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/index_human_robot_cfg5.py`. Script sau ghi lại CSV/JSON metadata nhỏ, không move/copy episode.
