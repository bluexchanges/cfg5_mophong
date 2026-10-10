# Báo cáo lượt 3 — Panda,3 chốt nhỏ và khuôn3 lỗ

**Báo cáo lịch sử trước rà soát placement.** Xem [PLACEMENT_LOGIC_REVIEW_VI.md](PLACEMENT_LOGIC_REVIEW_VI.md) cho config alias, độ nhô L−depth, support đáy, tests và kết quả evaluator đã sửa. Số14 tests và video dưới đây thuộc lượt3 ban đầu.

Đã dựng và chạy mô phỏng theo yêu cầu mới. Task thay khối hộp đề xuất cũ bằng **3 chốt trụ Ø6mm, dài35mm** và **3 lỗ Ø7mm, sâu12mm**, khớp màu đỏ/xanh lá/xanh dương. Lỗ cỡ bút chì;3 chốt bắt đầu dựng đứng trên bàn.

## ACT nào đang dùng?

Lựa chọn implementation là **[rh20t/act_baseline](https://github.com/rh20t/act_baseline/tree/107ce1287a46be3a1a7ea954f8bbdec3bb103288)**, commit đã khóa, với adapter CFG5/base-frame8D và8 RGB views viết ngoài upstream. ACT gốc và AirExo ACT-in-the-wild được khảo sát để đối chiếu; AirExo chưa là implementation chính. **Chưa tích hợp/chạy model forward hoặc training ACT.** Lượt3 dùng controller IK được lập trình thủ công; không có checkpoint trong demo.

## Kết quả chạy thật

| Kiểm tra | Kết quả |
|---|---|
| Model/engine | Panda Menagerie Apache-2.0,71 file tải riêng; MuJoCo3.14.0 |
| Robot | 7 arm joints,2 finger joints,8 actuators; kiểm names/limits/movement/gripper/reset/close |
| Scene | XML tải được,3 peg free bodies, cavity thật từ32 convex wedges/lỗ; test rơi vào đáy lỗ và nằm trên mặt khuôn |
| Camera | 8 policy view `(8,360,640,3)` uint8;6 fixed,2 wrist;2 debug cameras ngoài policy |
| Projection | Render marker đối chiếu K của8 camera, max error0.2213pixel |
| GUI/headless | Viewer mở/sync/đóng sau2s; offscreen RGB và video chạy được |
| Unit tests | 14/14 đạt:8 dữ liệu +6 mô phỏng |
| Controller smoke | Seeds42,43,44:3/3 episode thành công, mỗi episode đủ3 vật đúng lỗ |
| Video seed42 | 49.7 giây simulation,10 FPS;3/3 grasp, lift, place; không ghi nhận dropped |

Seed42 radial center errors cuối khoảng0.483/0.492/0.492mm, nằm trong clearance0.5mm; chốt sát thành, không nằm chính giữa tuyệt đối. Evaluator vẫn yêu cầu containment toàn bound của cylinder, height, tilt, release và ổn định0.5s. Margin phụ50µm thử ban đầu đã bỏ vì tiếp xúc thành là trạng thái placed hợp lệ; đường kính lỗ không được tăng để đạt kết quả. Kiểm tra chi tiết/lý do trong [hướng dẫn](MUJOCO_PANDA_GUIDE_VI.md).

3/3 là **smoke kỹ thuật** với jitter reset±1mm và scripted controller biết pose vật; không phải kết quả ACT hay bằng chứng độ tin cậy trên các pose/điều kiện tùy ý.

## File chính

- `configs/mujoco/panda_three_pegs.json`: cấu hình chính vật/khuôn, controller, physics, camera và evaluator. `precision_pick_place.json` cũ đã đánh dấu superseded.
- `mujoco/models/franka_emika_panda/`: upstream XML/license/manifest, asset ignored khỏi Git. `mujoco/scenes/panda_three_pegs.xml`: bản dẫn xuất có SHA/provenance.
- `src/simulation/panda_env.py`, `panda_controller.py`, `panda_evaluator.py`, `panda_demo.py`, `panda_observation.py`, `scene_builder.py`: API, IK, contact/geometry metrics và adapter.
- `scripts/download_panda_model.py`, `build_panda_scene.py`, `check_mujoco.py`, `run_panda_viewer.py`, `run_pick_place_demo.py`, `evaluate_mujoco.py`, `example_panda_env.py`: entrypoints đã có thật.
- `tests/test_simulation.py`, `requirements-sim-lock.txt`, [MUJOCO_PANDA_GUIDE_VI.md](MUJOCO_PANDA_GUIDE_VI.md).
- `docs/audit/mujoco_check.json`, `stage3_verification.json`: bằng chứng kiểm tra nhỏ.

Video: `outputs/three_pegs_demo/controller_demo.mp4`. Cùng folder có result.json, trace.json và final_closeup.png. Kết quả multi-seed tại `outputs/controller_evaluation/`. Ảnh8 camera và overview tại `outputs/stage3/`. Raw CFG5 và source ACT/Panda upstream giữ nguyên; chưa tạo Git repo/commit.

## Lệnh chạy

```powershell
Set-Location 'D:\Download\mophong_RH20T'
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/run_pick_place_demo.py --viewer --video --output outputs/my_demo
```

Mong đợi controller gắp từng chốt và đưa vào lỗ cùng màu; log có `Task success: True`. Để chỉ xem scene, chạy `scripts/run_panda_viewer.py`. Xem đầy đủ cài đặt/API/headless/troubleshooting ở [hướng dẫn MuJoCo](MUJOCO_PANDA_GUIDE_VI.md).

## Giới hạn và bước tiếp

Tool site ảo tại grasp center Menagerie có mapping rõ trong config nhưng chưa được đối chiếu EE firmware/offset thực tế CFG5. Rig camera giữ calibration tương đối theo TCP ảo; không tuyên bố đã tái tạo chính xác phần gá cơ khí của robot thật. IK không có planner tránh va chạm tổng quát; các đường đi demo đã được kiểm thử. Chốt bắt đầu đứng, không có task dựng từ nằm ngang.

Chưa có demonstrations RGB để train task đích, chưa chạy ACT hoặc training dài. Lượt4 sẽ thu demonstrations cùng contract, tích hợp RH20T ACT baseline, pretraining/fine-tuning, checkpoint/resume và offline/rollout evaluation. Không cần chốt lại lựa chọn kỹ thuật trong phạm vi đã giao.
