# Kiểm thử và ý nghĩa của từng file

Tests dùng `unittest`; một phần dùng dữ liệu giả nhỏ, một phần khởi tạo MuJoCo và dùng model Panda đã có. Đây là kiểm code/hình học/vật lý cơ bản, không đo năng lực ACT.

| File | Nhóm lỗi được kiểm | Phụ thuộc |
|---|---|---|
| `test_data_pipeline.py` | Chọn timestamp causal, cắt gap/chunk, quaternion, group split và contract, preprocessing+train-only statistics, số frame video, camera fixed/wrist. | NumPy/OpenCV và fixture tạm; không decode toàn CFG5. |
| `test_simulation.py` | Model/names/limits/reset/close, robot/gripper chuyển động, action validation, cavity vật lý thật, lực đỡ đáy, camera theo kinematics, evaluator và lịch sử. | MuJoCo và Panda model/assets cục bộ. |
| `test_placement_logic.py` | Alias,25−10=15 và35−12=23, biên tolerance, penetration, nghiêng/đổ/sai lỗ, floating không support, robot contact yếu, timer liên tục, config không hợp lệ và bounds đế. | Geometry/evaluator/config NumPy. |
| `__pycache__/` nếu có | Bytecode Python tự sinh. | Không sửa tay, không phải test/source cần giữ. |

Chạy từ project root:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -p test_placement_logic.py -v
```

Lần kiểm chứng logic placement ngày2026-09-27:27 tests đạt, xem [log](../docs/audit/placement_unit_tests.txt). Đây là kết quả lần chạy được lưu; khi đổi code phải kiểm lại phần liên quan. README mới không có nghĩa các tests đã được chạy thêm một lần.

Một số test đặt trực tiếp trạng thái giả để tạo tình huống thất bại có chủ đích. Cách tạo fixture đó không phải hành vi của demo controller. Demo3 seed và render8 camera là các kiểm chứng riêng, xem [audit](../docs/audit/README.md).

Khi sửa evaluator, cần giữ các ca vật đúng chiều cao nhưng nằm ngang/lơ lửng/còn chạm robot. Khi sửa dataset, giữ kiểm chống nhìn tương lai, chống nối gap và train-only normalization. Không đổi expected result chỉ để làm test xanh khi chưa hiểu tiêu chí task.

Bổ sung lượt4a: `test_act_integration.py` kiểm ACT (source hash, denominator, padding invariance, batch/hooks/state_dict roundtrip, camera identity/backward, contract); `test_data_selection.py` kiểm allowlist/provenance/hash/ID sai. Con số35 tests trong [log lịch sử](../docs/audit/act_unit_tests.txt) là trước khi thêm trainer. `test_pretraining.py` kiểm config toàn CFG5, sampler theo episode, metric weighted theo valid action, optimizer step thật với ACT thu nhỏ, checkpoint/resume và từ chối contract sai. Tests ACT/trainer cần Torch; môi trường chỉ dữ liệu sẽ skip các test ACT cũ nếu chưa có Torch.
