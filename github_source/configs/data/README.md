# Config dữ liệu CFG5

`cfg5_base8.json` là config lịch sử của subset5 task/57 episode. `cfg5_all8.json` giữ cùng representation, 8 camera và tiêu chí kỹ thuật, nhưng `task_ids: []` nghĩa là xét **mọi task CFG5**. Manifest/split mới là `data/splits/cfg5_all_manifest.json` và `cfg5_all_splits.json`; chưa có preprocessing toàn bộ. Các giá trị dưới đây chung cho hai config trừ phạm vi task.

| Nhóm trường | Logic và ý nghĩa |
|---|---|
| `dataset_root`, `task_ids`, `min_rating` | Giữ raw ở `RH20T_cfg5`; chọn task4/8/13/14/17 và rating≥2. Đây là subset pretraining, không phải toàn bộ RH20T. |
| `camera_serials`, `require_all_cameras` | Bắt buộc đủ8 RGB; giữ đúng thứ tự serial trong array ảnh. Episode thiếu camera bị loại, không điền ảnh giả. |
| `state_camera_serial` | Chọn nhánh timestamp/state đã transformed trong raw; state vẫn biểu diễn TCP trong aligned base frame. |
| `in_hand_serials` | Hai camera gắn tay:104422070042 và135122079702; sáu camera còn lại cố định. |
| `image_width`, `image_height` | Resize RGB về640×360. K camera cũng phải scale tương ứng. |
| `control_hz`, `max_observation_age_ms`, `max_state_age_ms` | Lưới10 Hz; lấy mẫu gần nhất ở quá khứ, tuổi≤150 ms. Không mượn ảnh/state tương lai. |
| `min_segment_steps` | Giữ đoạn liên tục có ít nhất3 bước; mất đồng bộ thì cắt segment. |
| `chunk_size` | Tối đa20 action tương lai, bắt đầu t+1; cuối segment được padding và mask. |
| `representation` | 8D=`x,y,z,qw,qx,qy,qz,width`; xyz/width theo mét. Arm target là pose đo được trong tương lai; gripper target dùng command width. |
| `gripper_scale_to_m`, `gripper_valid_range_m` | Chuyển width bằng hệ số0.001; kiểm miền0..0.085 m của dữ liệu. Giới hạn gripper mô phỏng được kiểm riêng. |
| `quaternion_norm_tolerance` | Loại quaternion raw có norm lệch quá0.05; geometry module chuẩn hóa và giữ dấu liên tục theo segment. |
| `split_group_fields`, `split_ratios`, `seed` | Nhóm theo(task,user,calib), tỷ lệ mục tiêu80/10/10, seed42; không chia frame ngẫu nhiên. Tỷ lệ episode thực tế có thể khác do chia theo group. |
| `normalization_std_floor` | Chặn std tối thiểu0.0001 để tránh chia cho0. Statistics chỉ fit trên train. |
| `image_normalization_mean/std` | Chuẩn hóa ImageNet sau khi RGB chia255. Adapter đã làm bước này; khi tích hợp model phải tránh làm lần hai. |

Kết quả cũ:57 episode được nhận, split47/5/5. Kết quả toàn CFG5:663 robot episode được nhận,658 robot bị loại,1.194 human được giữ riêng với chỉ mục; xem [báo cáo mới](../../docs/CFG5_FULL_FILTER_VI.md). Hash được tính từ nội dung config để phát hiện việc trộn artifact khác cấu hình. Xem [data/README](../../data/README.md) và [code dữ liệu](../../src/data/README.md).
