# Logic mô phỏng Panda và3 chốt

Package này mô phỏng robot, camera và task; nó chưa chứa ACT model. Đơn vị vật lý là mét, giây, newton; quaternion wxyz, các góc theo radian trừ trường có hậu tố `_deg`.

| File | Chức năng và quan hệ |
|---|---|
| `__init__.py` | Đánh dấu package. |
| `config_validation.py` | Kiểm schema2, kích thước/tolerance, lực, timing, workspace, vị trí lỗ/vật, jitter. Từ chối cấu hình mâu thuẫn trước khi build. |
| `task_geometry.py` | Công thức dùng chung: đáy trong=table+đế, miệng=đáy+depth, tâm chốt seated=đáy+L/2, độ nhô=L−depth; bounds đế theo offsets x/y. |
| `camera_rig.py` | Đọc calibration và config robot RH20T; đổi frame CV sang MuJoCo, scale K, xuất rig và tính pose camera theo base/TCP. |
| `scene_builder.py` | `load_sim_config` giải alias và validate. `build_scene` dẫn xuất Panda XML, dựng bàn/chốt/thành lỗ rỗng, gắn camera/TCP, ghi scene và manifest hash. |
| `panda_env.py` | `PandaEnv` quản lý model/data, reset, action step, vật lý/contact, observation, evaluator, renderer/viewer và đóng tài nguyên. Tự rebuild scene nếu các hash đang kiểm không khớp. |
| `panda_controller.py` | Kiểm action8D, normalize quaternion, giới hạn workspace/width; IK damped least squares → joint targets; giới hạn bước dịch/vận tốc khớp và đổi width sang actuator gripper. |
| `panda_observation.py` | `to_act_observation`: kiểm order/shape camera, normalize ảnh/state. `denormalize_action`: đưa output8D về đơn vị vật lý; controller tiếp tục kiểm miền hợp lệ. |
| `panda_evaluator.py` | `placement_geometry` kiểm hình học tức thời; `PandaEvaluator` lưu lịch sử grasp/lift, timer placement, drop và failure reasons của từng vật. |
| `panda_demo.py` | Controller theo các pha tiếp cận, hạ, kẹp, nâng, chuyển, cắm, nhả/rút. Dùng trạng thái vật đặc quyền và grip offset đo được để tạo action; không teleport vật trong demo. |

## Luồng thực thi

`config + rig + upstream XML → scene → PandaEnv.reset → observation → action → controller → physics/contact → evaluator → observation/info`.

`reset(seed)` trả `(obs, info)`. `step(action)` trả `(obs, reward, terminated, truncated, info)`; đây là API riêng có chữ ký tương tự Gymnasium, chưa đăng ký Gymnasium Env. Hết episode cần reset trước khi step tiếp. Dùng context manager hoặc `close()` để giải phóng renderer/viewer.

Action là pose TCP tuyệt đối+width, không phải8 actuator commands. Physics1000 Hz, evaluator100 Hz, action10 Hz. `obs.images` có shape(8,360,640,3) uint8; `obs.state` shape(8,) float32. Khi tắt render, images=None và không dùng được adapter ảnh ACT. `info.privileged_object_states` chỉ phục vụ debug/controller, không đưa vào input policy.

Reward bằng số vật `place_success` chia3; `terminated` cần `task_success` của cả3 vật, bao gồm lịch sử grasp/lift. `truncated` khi hết thời gian mà chưa thành công. Vì vậy reward cao không tự chứng minh đủ mọi điều kiện task.

## Điều kiện đặt đúng cần giữ

Độ nhô hiện tại35−12=23 mm, tolerance±0.5 mm. Kiểm thêm phần chốt trong đoạn đáy..miệng lỗ, góc nghiêng≤3°, sai số đáy, lực đỡ từ đáy≥0.001 N, không chạm robot, vận tốc nhỏ và ổn định liên tục0.5 s. Timer chỉ tính khoảng giữa hai mẫu cùng đạt. Không buộc phần nhô phía trên nằm trong hình chiếu khẩu độ; containment đang dùng bound bảo thủ.

Contact grasp cần đủ lực ở hai ngón; released kiểm mọi contact chạm/xuyên với robot, kể cả lực yếu. Lực đỡ chỉ tính thành phần+Z từ geom đáy khuôn. Chiều cao đúng nhưng lơ lửng, nghiêng quá mức hoặc sai lỗ vẫn fail. Xem [công thức đầy đủ](../../configs/mujoco/README.md).

TCP ảo ở grasp center Menagerie là quy ước mô phỏng, chưa phải factory EE transform của CFG5 được đo xác nhận. Các thay đổi frame/camera cần kiểm lại projection và kinematics. Giữ nguyên XML/model upstream; sửa builder/config rồi regenerate. Kiểm hồi quy ở [tests](../../tests/README.md).
