# Setup 8 camera theo CFG5

Trạng thái cập nhật lượt3: đã dựng/render8 policy views trong MuJoCo, kiểm fixed/wrist kinematics và projection marker (sai số lớn nhất0.222pixel). Xem [hướng dẫn mô phỏng](MUJOCO_PANDA_GUIDE_VI.md), gồm giới hạn của TCP ảo so với EE vật lý CFG5. Phần dưới ghi nguồn/công thức calibration từ lượt2.

Người dùng yêu cầu 8 camera như setup thu dữ liệu. Dùng calibration `1640071679624`, phổ biến nhất trong toàn bộ robot CFG5 đã kiểm kê (906/1321 episode). Sáu camera đứng cố định trong robot base; hai camera còn lại đi theo TCP. Không thay setup bằng tám camera cố định xếp quanh bàn.

| Thứ tự input | Serial | Gắn vào | Vị trí camera tại calibration trong aligned base (m), làm tròn |
|---:|---|---|---|
| 0 | 036422060215 | Robot base | −0.0800, 0.4445, 0.7887 |
| 1 | 037522062165 | Robot base | 0.8525, 0.2147, 0.8158 |
| 2 | 104122061850 | Robot base | 0.8463, −0.4317, 0.8141 |
| 3 | 104122063678 | Robot base | 0.3138, −0.5398, 0.7986 |
| 4 | 104422070042 | Aligned TCP | 0.4496, 0.0331, 0.3888 |
| 5 | 104422070044 | Robot base | −0.0962, −0.5297, 1.0175 |
| 6 | 105422061350 | Robot base | 0.3588, 0.4487, 0.7383 |
| 7 | 135122079702 | Aligned TCP | 0.5328, 0.0407, 0.3908 |

Nguồn chính xác, gồm **rotation**, K và SHA-256: `configs/mujoco/cfg5_camera_rig.json`. Không dùng riêng bảng xyz để dựng camera nhìn-at tùy ý. Serial `f0461559` không thuộc 8 RGB view được chọn.

## Quy ước toán học

`T_A_from_B` biến tọa độ trong B sang A. Mã nguồn RH20T quy định preprocessing TCP Franka: biến raw rotation vector thành ma trận rồi nhân `diag(1,−1,−1,1)`. Từ extrinsics/calibration TCP và tc_mat của config5 suy ra marker→base, sau đó camera→aligned base. Implementation theo công thức `RH20TScene.extrinsics_base_aligned`; không tự ước lượng bằng vị trí nhìn từ ảnh.

Đối với camera cố định, lưu mount `T_base_from_camera`. Với camera wrist, tại calibration tính:

`T_tcp_from_camera = inverse(T_base_from_tcp) @ T_base_from_camera`.

Trong mô phỏng camera wrist tại mỗi bước là `T_world_from_base @ T_base_from_tcp(t) @ T_tcp_from_camera`. Camera cố định không đổi khi TCP di chuyển. Cần gắn đúng **RH20T aligned TCP** vào Panda, không mặc định đó là body hand hoặc site mặc định của model.

Camera CV nhìn theo +Z, +Y hướng xuống; camera MuJoCo nhìn theo −Z, +Y hướng lên. Đổi trục bằng `diag(1,−1,−1,1)` trước khi gắn mount MuJoCo. JSON đã có `T_parent_from_camera_mujoco`.

## Intrinsics và ảnh

Intrinsics RH20T được hiệu chuẩn ở 1280×720; RGB tải xuống là 640×360. Nhân hai hàng đầu K với0.5 cho kích thước này, giữ homogeneous row. File K gốc có shape3×4, phần3×3 được dùng cho pinhole RGB. Không tự đặt fovy chung cho mọi camera.

Khi tạo XML ở lượt3, dùng fx/fy và cx/cy của mỗi camera. Theo API MuJoCo, `principalpixel` là offset so với tâm ảnh, không phải cx/cy tuyệt đối; cần chuyển phù hợp và kiểm chứng projection bằng render. `sensorsize`/focalpixel và phiên bản engine phải được kiểm tra trước khi xuất XML. Chưa mô phỏng distortion, ánh sáng, texture hay sensor noise của camera thật.

Hai calibration được giữ trong subset pretraining; rig mô phỏng ban đầu lấy một calibration. Vì vậy mô phỏng chưa phải bản sao toàn bộ điều kiện camera trong mọi episode CFG5.

## Kiểm chứng đã chạy

Chạy từ root:

```powershell
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/export_camera_rig.py
& '.\.venv-data\Scripts\python.exe' -X utf8 scripts/audit_camera_rig.py
```

Output `docs/audit/camera_rig_check.json`: 8 roundtrip tại calibration; kiểm tra 6 fixed không đổi/2 wrist đổi theo TCP; 12 camera/calibration crosschecks ×5 pose =60 phép đối chiếu `tcp_base.npy`→`tcp.npy` trên train. Không áp extrinsic wrist tại calibration cho toàn trajectory wrist chuyển động. Chưa xác minh dynamic wrist extrinsics bằng dữ liệu độc lập hoặc hình ảnh render.

Nguồn: [RH20T scene.py](https://github.com/rh20t/rh20t_api/blob/aa3124434729ed622109a29b2cbb9f3bbb1c5eeb/rh20t_api/scene.py), [RH20T configuration](https://github.com/rh20t/rh20t_api/blob/aa3124434729ed622109a29b2cbb9f3bbb1c5eeb/configs/configs.json), [ghi chú resize intrinsics](https://github.com/rh20t/rh20t_api/blob/aa3124434729ed622109a29b2cbb9f3bbb1c5eeb/README.md), [MuJoCo camera](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera).
