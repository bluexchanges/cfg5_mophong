# Cấu hình của dự án

Thư mục này quyết định dữ liệu nào được dùng, policy nhìn thấy gì và thế nào là đặt vật thành công. Code đọc JSON; README giải thích ý nghĩa, không thay thế JSON.

| File | Làm gì và ai đọc? | Điểm quan trọng |
|---|---|---|
| [data/cfg5_base8.json](data/cfg5_base8.json) | `src/data/common.py`, pipeline CFG5 và camera exporter đọc bộ lọc task, camera, thời gian, representation, split, normalization. | Thứ tự 8 camera và vector 8D là hợp đồng chung của dữ liệu và mô phỏng. |
| [mujoco/panda_three_pegs.json](mujoco/panda_three_pegs.json) | Config task chính: Panda, bàn, chốt, khuôn, controller, evaluator, render, đường dẫn model/rig. | Chốt Ø6×35 mm, lỗ Ø7 mm sâu12 mm; độ nhô mục tiêu35−12=23 mm, tolerance±0.5 mm. |
| [mujoco/precision_pick_place.json](mujoco/precision_pick_place.json) | Alias tương thích tới config chính, được `load_sim_config` giải quyết. | Không thêm kích thước hoặc threshold vào alias; không phải task hộp thứ hai. |
| [mujoco/cfg5_camera_rig.json](mujoco/cfg5_camera_rig.json) | Kết quả xuất calibration: camera order, intrinsics K, transform, parent frame, hash nguồn. Scene builder dùng để gắn 6 camera cố định và2 camera trên tay. | Đây là dữ liệu được sinh từ calibration CFG5, không phải các góc quay đoán bằng mắt. |

Luồng dữ liệu: `cfg5_base8.json → manifest → split → processed → normalization → ACTDataset`. Luồng mô phỏng: `panda_three_pegs.json + camera rig + Panda XML → scene builder → PandaEnv → evaluator`.

Khi sửa task, bắt đầu từ `panda_three_pegs.json`; loader kiểm quan hệ kích thước/timing trước khi tạo scene. Khi sửa camera hoặc representation của dữ liệu, phải kiểm lại hash của manifest, split, processed và statistics; đổi JSON không tự chuyển đổi các output cũ. Không sửa raw CFG5 để khớp config.

Đọc tiếp [giải thích config dữ liệu](data/README.md) và [công thức đặt vào khuôn](mujoco/README.md). Đã có [act/](act/README.md) với cấu hình kiến trúc/model smoke; chưa phải job training hoàn chỉnh.

Config model mới `act/rh20t_base8.json`: kiến trúc RH20T ACT8 camera/8D/K20, pretrained weights=null, thông số loss/optimizer. Đã kiểm inference, chưa training.
