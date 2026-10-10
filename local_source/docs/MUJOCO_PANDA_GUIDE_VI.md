# Panda và khuôn 3 lỗ trong MuJoCo

Lượt3 đã có mô phỏng chạy thật, controller và evaluator. Mục tiêu hiện tại: gắp **3 chốt trụ Ø6 mm, dài35 mm**, đưa từng chốt vào lỗ cùng màu **Ø7 mm, sâu12 mm**. Đường kính lỗ cỡ bút chì là lựa chọn thiết kế theo yêu cầu, không khẳng định mọi loại bút chì đều cùng kích thước. Khoảng hở bán kính0.5 mm; không còn dùng khối hộp30 mm của đề xuất lượt2.

## 1. Model, engine và phạm vi

MuJoCo tính động lực học, va chạm và ma sát. Panda có7 revolute joints và2 finger slide joints, điều khiển bằng7 arm actuators và1 gripper tendon actuator. Peg là free body chịu trọng lực, không weld/teleport/attach vào gripper trong demo. Robot base cố định vào world; root của robot là aligned base dùng cho action/camera.

Nguồn: [MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/franka_emika_panda), Apache-2.0. Commit `c96a32d28fb5da84da38c1da4d749e7a13212855`;71 file XML/license/mesh với SHA-256 trong `mujoco/models/franka_emika_panda/manifest.json`. Script tải riêng Panda, không clone toàn Menagerie. Mesh asset được ignore khỏi Git, manifest/XML/license được giữ. Upstream XML không sửa; scene là bản dẫn xuất được tạo bởi scene_builder.

Đã kiểm trên Windows CPython3.12.14, MuJoCo3.14.0 và các dependency khóa trong `requirements-sim-lock.txt`. Không cài Torch/checkpoint, chưa tích hợp hoặc chạy ACT. Viewer và renderer dùng native bindings [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html). Sau rà soát logic, config schema2 tính trực tiếp độ nhô mục tiêu; xem [sửa lỗi placement](PLACEMENT_LOGIC_REVIEW_VI.md).

## 2. Chạy từ đâu và cài môi trường

Mở PowerShell ở project root:

```powershell
Set-Location 'D:\Download\mophong_RH20T'
& '.\.venv-data\Scripts\python.exe' --version
& '.\.venv-data\Scripts\python.exe' -m pip check
```

`.venv-data` hiện đã có dependency mô phỏng. Không dùng Python MSYS2 trên PATH lẫn với môi trường này. Nếu tái tạo trên máy khác, tạo venv bằng standard Windows CPython3.12, rồi:

```powershell
& '.\.venv-data\Scripts\python.exe' -m pip install -r requirements-sim-lock.txt
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/download_panda_model.py
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/build_panda_scene.py
```

Tải model cần mạng lần đầu; khi đủ file thì downloader kiểm SHA và chạy offline. Thiếu/corrupt asset sẽ báo lỗi, không thay nguồn bằng model giả. Input builder là `configs/mujoco/panda_three_pegs.json`, rig CFG5, XML/mesh upstream. Output là `mujoco/scenes/panda_three_pegs.xml` và manifest nguồn/hash. PandaEnv tự rebuild khi config, source XML, rig hoặc generator đổi. Không sửa tay generated XML để làm thí nghiệm; chỉnh config/builder rồi tạo lại.

## 3. Kiểm tra scene và 8 camera

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/check_mujoco.py
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
```

Kỳ vọng MuJoCo3.14.0, RGB `(8,360,640,3)` uint8; bộ tests gồm dữ liệu, mô phỏng và hồi quy placement. Output `docs/audit/mujoco_check.json`, ảnh tại `outputs/stage3/`. `eight_cameras.png` là contact sheet, mỗi view ghi serial; `overview.png` và `task_closeup.png` để xem scene.

8 observation cameras giữ thứ tự/calibration của lượt2:6 fixed và2 wrist. Hai camera debug `overview`, `task_closeup` không được đưa vào ACT, nên model XML có10 cameras nhưng observation vẫn đúng8. Test xác nhận camera wrist đi theo TCP thật và camera fixed đứng yên.

K640×360 lấy từ calibration1280×720 nhân0.5. Camera CV→MuJoCo đổi trục Y/Z. `principalpixel` trong XML dùng `[width/2-cx,height/2-cy]`; đã kiểm bằng render segmentation marker độc lập, sai số centroid lớn nhất0.222 pixel. Không chỉ kiểm ma trận khớp chính nó. Không giả lập lens distortion/noise/texture của sensor thật.

**TCP mapping:** site ảo `policy_tcp` tại hand+[0,0,0.1034]m, giữ axes Menagerie hand. Đây là frame policy được định nghĩa rõ trong mô phỏng; pose/state theo base với quaternion wxyz. Không khẳng định offset/axes giữa EE firmware của robot thật CFG5 và Menagerie hand đã được hiệu chuẩn độc lập. Khi replay trực tiếp TCP dữ liệu thật hoặc chuyển ra robot thật phải xác minh tool transform này. Pretraining/fine-tuning dùng cùng contract8D; chưa chứng minh sim-to-real.

## 4. Viewer và demo

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/run_panda_viewer.py
```

Mở scene tĩnh; đóng cửa sổ để thoát. Đã kiểm launch_passive/sync/close bằng chế độ `--seconds 2`. Muốn xem robot thực hiện task:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/run_pick_place_demo.py --viewer --video --output outputs/my_three_pegs_demo
```

Không cần viewer để chạy physics/render video:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/run_pick_place_demo.py --video --output outputs/my_headless_demo
```

Output: result.json (config, seed, stages, metrics), trace.json (actions/TCP/object positions/contact từng control step), controller_demo.mp4 (10 FPS), final_closeup.png. Chọn output folder riêng cho mỗi thí nghiệm; script có thể ghi đè kết quả cùng tên. Raw CFG5 bị chặn làm output.

Đã chạy seed42: đặt3/3 chốt,49.7 giây simulation. Video nằm trong `outputs/three_pegs_demo/controller_demo.mp4`. Mô phỏng chạy nhanh hơn wall-clock tùy máy; `--viewer` hiện sync từng bước nhưng không cưỡng bức realtime. Video vẫn có nhịp10 FPS.

Demo dùng **privileged state** của vật để chọn waypoint và đo offset sau gắp. Không có ACT/checkpoint trong demo. Chốt được kẹp bằng tiếp xúc/ma sát; chỉ reset và fixture unit tests mới đặt qpos trực tiếp. Kết thúc ngay khi đủ3 chốt đạt tiêu chí; tay có thể còn phía trên chốt cuối sau khi mở, vì retreat không là điều kiện thành công.

## 5. API dùng từ Python

```python
from src.simulation.panda_env import PandaEnv

with PandaEnv(config_path="configs/mujoco/panda_three_pegs.json") as env:
    obs, info = env.reset(seed=42)
    action = obs["state"].astype(float).copy()
    action[0] += 0.01  # target TCP x tăng 1 cm
    action[7] = 0.04  # width 4 cm
    obs, reward, terminated, truncated, info = env.step(action)
    rgb = env.render("overview")
```

Chạy ví dụ có sẵn: `& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/example_panda_env.py`. Đây là API riêng có reset/step kiểu5-tuple; chưa đăng ký Gymnasium/spaces. Không gọi Gymnasium wrapper như thể env đã là Gymnasium Env.

`reset(seed)` trả trạng thái sau0.25s ổn định; jitter vị trí chốt ±1mm tái lập theo seed. `step(action)` nhận vector vật lý8D `[x,y,z,qw,qx,qy,qz,width_m]`; action không phải tensor chuẩn hóa hay actuator ctrl. Hàm kiểm NaN/Inf, quaternion khôngzero, workspace, width0–80mm; quaternion được normalize. Kết thúc vì success trả terminated=True, hết70s trả truncated=True. Sau đó phải reset trước step tiếp.

Physics1000Hz, policy10Hz, evaluator100Hz. Controller dùng damped least-squares IK, giới hạn bước TCP và tốc độ khớp rồi gửi position targets; có gravity compensation cho robot. Giới hạn lực gripper20N, ma sát finger theo config. Không có planner tránh va chạm tổng quát hoặc đảm bảo mọi pose trong hộp workspace đều khả thi; info.controller báo residual IK để kiểm tra. Scripted waypoints là đường đi đã smoke test.

Observation: images8 RGB và state8D float32. Quaternion state chọn dấu liên tục qua thời gian. `info.privileged_object_states` dành cho debug/controller/evaluator, không là input ACT. Đặt `render_observations=False` khi chỉ test physics; images=None và adapter ACT sẽ từ chối. `to_act_observation` chuyển NHWC uint8→NCHW float32/ImageNet và normalize state với statistics được cung cấp; không fit statistics trong simulation. `denormalize_action` khôi phục đơn vị trước `env.step`. Kiểm tra/checkpoint statistics thuộc lượt4.

`env.render(camera)` trả RGB; `env.object_states()` đọc pose/speed; `env.finger_contacts()` đọc tiếp xúc và lực thật từ `mj_contactForce`. `env.close()` giải phóng renderer/viewer, gọi lặp lại được; context manager tự đóng cả khi có exception.

## 6. Khuôn và định nghĩa thành công

Mỗi lỗ là32 convex mesh wedges ghép quanh khoảng rỗng, trên base dày4mm. Không dùng một concave mesh rồi để convex hull lấp lỗ. Apothem của polygon là3.5mm;3 chốt radius3mm. Unit test thả fixture vào lỗ thì dừng ở đáy; fixture ở mặt khuôn thì dừng cao hơn12mm.

Mỗi chốt phải vào **đúng lỗ cùng chỉ số/màu**, không chỉ một lỗ bất kỳ. Grasp cần tiếp xúc với cả hai finger body, mỗi contact normal force>0.005N trong0.1s. Lift cần đã grasp, đang giữ và tâm vật tăng≥20mm so với reset.

Độ nhô thực tế=`z_top-z_mouth`, mục tiêu=`peg_length-hole_depth=35−12=23mm`; sai số cho phép±0.5mm. Tilt≤3°, speed≤5mm/s và angular speed≤0.1rad/s. Chỉ phần cylinder giao khoảng đáy..miệng lỗ phải nằm trong khẩu độ: evaluator lấy bound bảo thủ theo hai đầu đoạn trục có thể đóng góp vào phần đó, cộng bán kính. Phần nhô phía trên không bị ép nằm trong lỗ. Còn phải có lực đỡ đáy≥0.001N, khe hở đáy≤0.5mm và lún số học≤0.1mm. Released cần không còn bất kỳ tiếp xúc robot nào, kể cả lực yếu; gripper đủ mở hoặc TCP đã rời xa. Các điều kiện giữ liên tục0.5s ở evaluator100Hz; không cộng thời gian trước mẫu đầu tiên. Công thức và ví dụ25−10=15 ở [config README](../configs/mujoco/README.md).

Cho phép chạm thành lỗ: `placement_margin_m=0`. Margin50µm thử ban đầu đã loại cả vật seated sát thành nhưng vẫn nằm trong lỗ; sau kiểm tra đã bỏ margin bổ sung, vẫn kiểm containment, height, tilt, release và stability. Không tăng đường kính lỗ hay nới height để đạt success. `object_dropped` chỉ ghi khi đã lift rồi mất giữ và về gần mặt bàn ngoài đúng lỗ; không nhầm chạm đáy khuôn thành rơi.

`task_success` mỗi vật yêu cầu lịch sử grasp+lift và place hiện tại; cả episode chỉ success khi cả3 đạt. Unit test đặt trực tiếp vật vào lỗ có thể cho place_success nhưng không task_success. Failure reasons và radial/height/tilt errors được lưu; không dựa vào nhận xét nhìn video.

## 7. Đánh giá nhiều seed và giới hạn

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/evaluate_mujoco.py --mode controller --seeds 42 43 44 --output outputs/my_controller_evaluation
```

Output episode_<seed>.json, episodes.csv, summary.json. Kiểm tra hiện tại3/3 episode thành công, mỗi episode đủ3 vật. Đây chỉ là **smoke kỹ thuật 3 seed**, không phải success rate ACT hoặc bằng chứng tổng quát hóa. CLI lượt3 chỉ hỗ trợ controller; lượt4 mới tích hợp model/checkpoint.

Giới hạn: chốt bắt đầu đứng sẵn, chưa học dựng chốt nằm ngang;3 chốt cùng hình dạng, phân biệt màu; chỉ jitter vị trí nhỏ; controller biết pose vật và lỗ; chưa có checkpoint ACT; chưa xác minh tool offset vật lý CFG5. Các thông số friction, gripper force, geometry, seeds, evaluation thresholds là config của task mô phỏng, không phải thông số đo thực tế.

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| No module named mujoco | Dùng đúng .venv-data và cài requirements-sim-lock.txt |
| Không tìm thấy mesh/manifest | Chạy downloader từ root; không chỉ copy XML |
| Hash model sai | Kiểm file nguồn bị sửa; phục hồi đúng commit, không bỏ kiểm SHA |
| OpenGL/GLFW lỗi | Cập nhật driver GPU, chạy trong Windows desktop có OpenGL; physics vẫn chạy với render_observations=False, nhưng chụp ảnh/video vẫn cần GL context |
| Action ngoài workspace hoặc width | Denormalize bằng statistics checkpoint, kiểm đúng mét và quaternionwxyz |
| TCP residual lớn | Target khó với IK, quá gần giới hạn hoặc đường đi va chạm; xem info.controller/trace, không teleport robot |
| RuntimeError sau success/timeout | Gọi reset trước episode mới |
| Chốt gắp/đặt không đạt sau đổi config | Xem contact, geometry errors và video; giữ evaluator rõ ràng, không gọi thất bại là success |

Bước tiếp theo là thu demonstrations đúng contract rồi tích hợp ACT/pretraining/fine-tuning ở lượt4. Model đã chọn là **RH20T ACT baseline**; ACT gốc và AirExo dùng để đối chiếu, chưa lấy AirExo làm code chạy chính.
