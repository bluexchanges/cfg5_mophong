# Sửa logic độ nhô và đặt vào khuôn

Đã rà cả3 JSON trong configs/mujoco và nơi sử dụng ở builder, demo, evaluator. Phản hồi của người dùng đúng về đại lượng cần nhìn: với vật25mm, lỗ sâu10mm, độ nhô mục tiêu là15mm khi đứng và chạm đáy. Tolerance0.5mm là sai số quanh15mm. Khi nghiêng/đổ, không được chỉ dựa trên chiều cao để kết luận placed.

## Lỗi đã sửa

| Trước | Sau |
|---|---|
| Config hộp cũ còn đầy đủ thông số bên cạnh config3 chốt, dễ bị dùng nhầm | precision_pick_place.json chỉ còn alias; loader resolve tới cấu hình chuẩn, cấm alias trộn thông số riêng |
| Height tolerance tên chung, không đo/ghi trực tiếp độ nhô | height_reference=peg_top_relative_to_hole_mouth; target suy ra L−depth; ghi expected/actual/error; tolerance±0.5mm |
| Kiểm containment cả chiều dài chốt, kể cả phần nhô phía trên lỗ | Chỉ kiểm bound bảo thủ của phần cylinder giao khoảng đáy..miệng lỗ; không ép phần nhô vào khẩu độ |
| Trong tolerance đáy + stable có thể được chấm placed dù chưa có lực đỡ | Bắt buộc contact đáy thật với lực đỡ hướng+Z ≥0.001N |
| Cùng một sai số đối xứng chấp nhận vật chìm xuống đáy | Tách max_bottom_gap_m=0.5mm khỏi max_floor_penetration_m=0.1mm |
| Dùng ngưỡng lực grasp để quyết định đã nhả, bỏ sót tiếp xúc yếu hoặc tiếp xúc phần khác của robot | Grasp vẫn cần lực đủ lớn; released dùng mọi tiếp xúc xuyên/chạm thực với robot, độc lập ngưỡng grasp |
| Mẫu đầu đã được cộng nguyên dt giữ/ổn định | Chỉ cộng khi điều kiện đạt ở hai mốc liên tiếp; mất điều kiện thì reset timer |
| Task fail do thiếu grasp/lift có thể trả failure_reason rỗng | Ghi missing_grasp/missing_lift cùng các lỗi placement |
| Loader chỉ kiểm số vật, bán kính, control timestep | Kiểm dimensions/tolerance/force/timing/layout/jitter và các quan hệ vật lý trước khi build |
| Đế khuôn giả định các lỗ đối xứng ở x=0 | Suy ra bounds đế từ offsets ở cảx/y, dùng chung helper |

## Kích thước không bị đổi để đạt success

Task vẫn là3 chốt Ø6×35mm,3 lỗ Ø7mm sâu12mm, đế4mm. Mục tiêu độ nhô **23mm**, dải **22.5–23.5mm**; đáy z4mm, miệng z16mm, đỉnh thẳng z39mm. Không tăng độ rộng lỗ hoặc bỏ kiểm góc để cho vật đổ vẫn thành công. Ví dụ25−10=15 được đưa vào unit test riêng.

Config/công thức giải thích tại [configs/mujoco/README.md](../configs/mujoco/README.md). `hole_depth_m` là độ sâu trong, không phải chiều cao khối khuôn gồm cả đế.

## Kiểm chứng

**27/27 unit tests đạt.** Headless render đủ8 camera đạt; sai số marker projection lớn nhất0.2213 pixel. Log: `docs/audit/placement_unit_tests.txt` và `docs/audit/placement_camera_check.json`.

Tests mới gồm:25−10=15,35−12=23; biên tolerance; lún xuyên đáy; chốt nghiêng hợp lệ với phần nhô ngoài khẩu độ; chốt nằm ngang dù đỉnh đúng target; đặt sai lỗ; lơ lửng không lực đỡ; tiếp xúc robot yếu; timer liên tục; cấu hình mâu thuẫn; đế dịch theo layout; alias không còn geometry riêng.

Đã chạy lại controller seeds42/43/44 qua chính đường dẫn alias người dùng phản ánh. Cả3 episode đặt đủ3 vật với evaluator mới. Seed42 đo độ nhô khoảng22.9989mm, sai số khoảng−0.0011mm so với23mm, lực đỡ mỗi chốt khoảng0.049N, phù hợp trọng lượng chốt5g. Log thực ở `outputs/placement_logic_review/episode_42.json` và summary.json. Đây vẫn là scripted controller smoke, chưa phải ACT.

Các báo cáo lượt3 cũ giữ làm lịch sử; kết quả sửa này ở `docs/audit/placement_logic_verification.json`. Dữ liệu CFG5, snapshot ACT và model Panda upstream không sửa.
