# Protocol pretraining CFG5 → đặt vật vào khuôn

Đây là thiết kế thí nghiệm, chưa có kết quả ACT. Mục tiêu kiểm chứng: pretraining CFG5 có giúp Panda đặt vật vào khuôn nhỏ chính xác hơn, hoặc cần ít demonstrations task đích hơn hay không?

## Baseline và dữ liệu

Hai nhánh bắt buộc:

1. **Target-only:** ACT được khởi tạo theo cùng quy tắc backbone, học demonstrations mô phỏng task đích.
2. **CFG5→target:** cùng kiến trúc, pretrain subset CFG5 rồi fine-tune trên đúng demonstrations task đích của nhánh1.

Giữ cùng camera order, input resolution, action representation, target split, số bước fine-tuning và model selection. Nếu dùng pretrained ImageNet backbone, cả hai nhánh dùng cùng weights; ghi rõ nguồn/hash. Nhánh2 có thêm ngân sách pretraining, phải báo tổng compute riêng. Thử ít nhất seeds42,43,44 khi chạy nghiên cứu chính; smoke không phải thí nghiệm nhiều seed.

CFG5 baseline: config `configs/data/cfg5_base8.json`, 5 task liên quan, 47/5/5 episode, group `(task,user,calib)`. Nhóm này không đảm bảo user/calibration độc lập toàn cục. Chỉ dùng train cho statistics; checkpoint pretraining chọn từ val CFG5. Test CFG5 chỉ đánh giá sau khi khóa cấu hình; lượt2 mới kiểm tính đầy đủ metadata/timeline, chưa tính loss hoặc chọn checkpoint theo test.

Task đích phải có demonstrations riêng thu trong MuJoCo sau khi controller/evaluator được kiểm thử. Tách episode và seed initial-state trước khi tạo chunks. Cùng một trajectory dù thay augmentation/camera vẫn cùng split. Không dùng success của scripted controller làm success ACT.

## Action và transfer normalization

Chọn pose aligned base xyz/wxyz + gripper width mét. Action là future pose proxy; không phải torque. Controller chuyển pose→Panda commands, kiểm quaternion, workspace, joint limit và gripper limit. Demo mô phỏng phải lưu cùng semantics cho cả hai nhánh.

Không đổi normalization rồi dùng nguyên head như thể không có thay đổi. Khi tích hợp training, phương án baseline là tiếp tục dùng statistics CFG5-train cho nhánh pretrained trong fine-tuning/inference; lưu statistics trong checkpoint. Nhánh target-only dùng statistics target-train. So sánh metric vật lý sau denormalize và rollout để tránh so normalized loss giữa hai bộ statistics khác nhau. Nếu chuyển cả nhánh pretrained sang target-train statistics phải viết migration/khởi tạo lại đầu vào-đầu ra có kiểm thử và ghi đó là ablation. Không fit statistics từ target val/test.

## Metric offline

Training loss gồm reconstruction L1 valid-normalized và KL với trọng số trong config. Khi chưa có training code, chưa chọn giá trị KL/LR/batch size theo phỏng đoán phần cứng. Validation dùng toàn bộ sample nếu có thể; nếu giới hạn thì config phải ghi giới hạn và seed, báo sample/batch/episode/valid-target count thực sự.

Tổng hợp L1 bằng tổng absolute error chia tổng valid scalar, không lấy trung bình không trọng số của batch có padding khác nhau. Metric phụ: position error (mm), quaternion geodesic error `2*acos(clamp(abs(dot(q_pred,q_true)),0,1))`, gripper width error (mm). Quaternion normalize trước metric; output invalid phải báo riêng, không âm thầm biến thành identity. Offline validation không thay thế rollout.

## Task khuôn và metric rollout

Config đang dùng `configs/mujoco/panda_three_pegs.json`:3 chốt Ø6×35 mm,3 lỗ Ø7 mm sâu12 mm, radial clearance0.5 mm. Config hộp cũ chỉ còn alias. Độ nhô mục tiêu23mm, tolerance±0.5mm; thêm kiểm khe hở đáy≤0.5mm, lún≤0.1mm và lực đỡ đáy≥0.001N. Containment cho phép chạm thành (margin0), tilt≤3°, ổn định0.5s. Đã sửa/kiểm physics/controller trước khi có ACT; không đổi threshold sau khi mở ACT test.

Đặt đúng kiểm phần cylinder trong khoảng đáy..miệng lỗ bằng bound bảo thủ, không bắt phần nhô phía trên nằm trong khẩu độ. Đồng thời kiểm độ nhô thực tế quanh target L−depth, seated/support, nhả robot và ổn định100Hz. Cylinder đối xứng quanh trục nên không chấm yaw. Grasp cần lực contact hai ngón>0.005N trong0.1s, lift≥20mm; stable linear/angular speed≤0.005m/s và0.1rad/s. Xem [định nghĩa và nguồn kiểm thử](MUJOCO_PANDA_GUIDE_VI.md). Cả3 vật phải đúng lỗ theo màu; direct fixture placement không đủ task_success nếu thiếu lịch sử grasp/lift.

Các metric riêng: grasp_success (tiếp xúc hai ngón và giữ), lift_success, place_success, task_success theo thứ tự trong cùng episode, object_dropped, episode_length, failure_reason. Báo success rate với tử/mẫu, seeds và interval bất định khi đủ số episode; không chỉ chọn video thành công. Tập rollout test dùng cùng initial conditions cho cả hai nhánh, chỉ chạy sau model selection. Headless/controller demo là kiểm kỹ thuật riêng.

## Truy vết và ablation

Checkpoint cần model/optimizer/scheduler, epoch/global step, random states Python/NumPy/Torch/CUDA, config/hash, split/hash, camera order, normalization, upstream commit và dependency versions. Resume phải kiểm contract. Chưa tạo checkpoint ở lượt2.

Ưu tiên kiểm chứng lợi ích pretraining trước. Các ablation sau có thể thay số demonstration đích, chunk size, subset task hoặc missing-view masking. Mỗi lần đổi một yếu tố, khóa val protocol trước khi mở test. Không dùng số liệu paper làm mục tiêu trực tiếp khi task, camera, split, embodiment hoặc metric khác nhau.

Giới hạn số57 episode trong bản protocol ban đầu thuộc subset5 task. Sau kiểm toàn CFG5 có663 robot episode đạt strict8-view filter trên92 task (xem [báo cáo](CFG5_FULL_FILTER_VI.md)); nhiều segment vẫn ngắn, task pretrain đa mục tiêu chưa có task-conditioning; chuyển miền thật→mô phỏng và khác biệt gripper; TCP proxy không chứa dynamics/force cần thiết cho tiếp xúc sát khuôn. Mọi khẳng định cải thiện cần kết quả so sánh thực tế.
