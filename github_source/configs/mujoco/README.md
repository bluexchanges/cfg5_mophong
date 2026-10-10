# Cấu hình nào đang chạy và thế nào là đặt đúng?

- **panda_three_pegs.json:** cấu hình duy nhất chứa kích thước/task/evaluator đang chạy.
- **precision_pick_place.json:** alias tương thích, tự chuyển tới panda_three_pegs.json khi đọc bằng load_sim_config. Không còn kích thước hộp25mm/khuôn10mm của đề xuất cũ.
- **cfg5_camera_rig.json:** calibration8 camera, không phải task config và không chứa tiêu chí placed.

## Độ nhô mục tiêu khác với tolerance

Vật đứng thẳng và chạm đáy:

`độ nhô mục tiêu = chiều cao vật - độ sâu lòng lỗ`

`độ nhô thực tế = z_đỉnh_vật - z_miệng_lỗ`

`đạt chiều cao ⇔ |độ nhô thực tế - độ nhô mục tiêu| <= tolerance`

Ví dụ vật25mm và lỗ sâu10mm: mục tiêu **15mm**. Tolerance0.5mm nghĩa là chấp nhận **14.5–15.5mm**, không phải yêu cầu vật chỉ nhô0.5mm.

Lưu ý: **độ sâu lòng lỗ** tính từ đáy trong tới miệng, khác chiều cao tổng khối khuôn nếu khối khuôn có đế dày. Nếu khối khuôn cao10mm gồm đế2mm thì lỗ chỉ sâu8mm, vật25mm nhô17mm. Không lấy chiều cao ngoài thay cho hole_depth_m.

## Bộ3 chốt hiện tại

| Đại lượng | Giá trị | Nguồn/công thức |
|---|---:|---|
| Chiều dài chốt | 35mm | peg_length_m=0.035 |
| Đường kính chốt/lỗ | 6/7mm | 2×peg_radius_m / 2×hole_radius_m |
| Độ sâu lòng lỗ | 12mm | hole_depth_m=0.012 |
| Bề dày đế | 4mm | mold_floor_thickness_m=0.004 |
| Mặt bàn | z=0mm | table_surface_z_m |
| Đáy trong lỗ | z=4mm | mặt bàn + bề dày đế |
| Miệng lỗ | z=16mm | đáy lỗ + độ sâu lỗ |
| Tâm chốt seated thẳng | z=21.5mm | đáy lỗ +35/2 |
| Đỉnh chốt seated thẳng | z=39mm | đáy lỗ +35 |
| **Độ nhô mục tiêu** | **23mm** | 39−16 =35−12 |
| Dải độ nhô cho phép | **22.5–23.5mm** | target ±protrusion_tolerance_m=0.0005 |

Độ nhô mục tiêu luôn được suy ra từ kích thước, không lưu thêm một hằng số23mm có thể lệch khi sửa chiều dài/độ sâu. Code dùng chung công thức ở `src/simulation/task_geometry.py` cho demo và evaluator.

## Vì sao đúng độ nhô vẫn chưa đủ?

Khi vật nghiêng hoặc đổ, chiều cao theo phương đứng thay đổi. Vật đặt nhầm lỗ, mắc ở miệng lỗ hoặc còn bị robot giữ cũng có thể tình cờ đạt chiều cao. Evaluator bắt buộc đồng thời:

1. Độ nhô thực tế đúng mục tiêu ±0.5mm.
2. Chốt nghiêng không quá3°; cylinder đối xứng nên không chấm yaw.
3. Phần vật **nằm giữa đáy và miệng lỗ** ở trong đúng lỗ cùng màu. Phần nhô phía trên không bị ép phải nằm trong hình chiếu của lỗ.
4. Đáy vật không cách đáy trong quá0.5mm và không lún số học sâu quá0.1mm. Hai giới hạn này tách biệt với độ nhô.
5. Có lực đỡ thực từ geom đáy khuôn ≥0.001N; vật lơ lửng hoặc mắc thành mà không chạm đáy không đạt.
6. Không còn tiếp xúc robot, kể cả tiếp xúc yếu chưa đạt ngưỡng grasp; gripper đã mở hoặc rời xa.
7. Vận tốc đủ nhỏ và giữ các điều kiện trên liên tục0.5s. Mẫu quan sát đầu tiên không tự được tính là đã giữ trong một khoảng thời gian trước đó.
8. Task success cần lịch sử grasp và lift, rồi place; cả3 vật phải đạt.

Config loader từ chối lỗ nhỏ hơn chốt, mục tiêu protruding mà vật ngắn hơn lỗ, tolerance đủ lớn để vật chưa cắm vẫn đạt, độ lún lớn hơn đế, cell khuôn chồng lấp, jitter chồng vật và timing không phù hợp physics timestep.

## Kiểm tra lại

Chạy ở project root:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/evaluate_mujoco.py --config configs/mujoco/precision_pick_place.json --seeds 42 43 44 --output outputs/placement_logic_recheck
```

Báo cáo sửa logic: `docs/PLACEMENT_LOGIC_REVIEW_VI.md`. Kết quả sau sửa ở `outputs/placement_logic_review/`; các báo cáo/video lượt3 trước đó là snapshot của evaluator cũ.
